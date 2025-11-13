#!/usr/bin/env python3
"""
AI Code Review Agent using Claude Sonnet 4.5
Analyzes PRs, posts reviews, and updates Jira tickets
"""

import os
import json
import anthropic
from jira import JIRA
import requests
from typing import Dict, List, Optional
import sys
import base64
import logging
import time
import re
from pathlib import Path
from urllib.parse import urlparse

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup structured logging with sensitive data masking."""
    
    # Create formatter that masks sensitive data
    class SensitiveDataFormatter(logging.Formatter):
        SENSITIVE_PATTERNS = [
            (r'sk-ant-[a-zA-Z0-9-]+', 'sk-ant-***MASKED***'),
            (r'token\s*[:=]\s*([a-zA-Z0-9_-]+)', r'token: ***MASKED***'),
            (r'api[_-]?key\s*[:=]\s*([a-zA-Z0-9_-]+)', r'api_key: ***MASKED***'),
            (r'password\s*[:=]\s*([^\s]+)', r'password: ***MASKED***'),
        ]
        
        def format(self, record):
            message = super().format(record)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
            return message
    
    logger = logging.getLogger('code_review_agent')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = SensitiveDataFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = SensitiveDataFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, anthropic.APIError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        # Add jitter
                        delay += (delay * 0.1 * (attempt + 1))
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logging.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


def validate_environment_variables(required_vars: List[str], logger: logging.Logger) -> Dict[str, str]:
    """Validate and return required environment variables with helpful error messages."""
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            logger.error(f"Missing required environment variable: {var}")
        else:
            env_vars[var] = value
            logger.debug(f"Found environment variable: {var}")
    
    if missing_vars:
        error_msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set these variables in your GitHub Actions secrets or environment."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return env_vars


def validate_file_path(filepath: str, logger: logging.Logger) -> bool:
    """Validate file path to prevent directory traversal and injection attacks."""
    try:
        # Resolve to absolute path
        abs_path = Path(filepath).resolve()
        
        # Check for directory traversal attempts (but allow absolute paths)
        if '..' in filepath:
            logger.warning(f"Potentially unsafe file path (contains ..): {filepath}")
            return False
        
        # Check if file exists
        if not abs_path.exists():
            logger.warning(f"File does not exist: {filepath}")
            return False
        
        # Check if it's actually a file (not a directory)
        if not abs_path.is_file():
            logger.warning(f"Path is not a file: {filepath}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating file path {filepath}: {e}")
        return False


def sanitize_input(text: str, max_length: int = 1000000) -> str:
    """Sanitize input to prevent injection attacks and limit size."""
    if not text:
        return ""
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
        logging.warning(f"Input truncated to {max_length} characters")
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Basic sanitization (can be extended)
    return text


def validate_diff_size(diff: str, max_size: int, logger: logging.Logger) -> bool:
    """Validate PR diff size before processing."""
    diff_size = len(diff.encode('utf-8'))
    
    if diff_size > max_size:
        logger.warning(
            f"Diff size ({diff_size} bytes) exceeds maximum ({max_size} bytes). "
            f"Review will be skipped."
        )
        return False
    
    logger.info(f"Diff size: {diff_size} bytes (max: {max_size} bytes)")
    return True


class CodeReviewAgent:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Code Review Agent with configuration."""
        # Setup logging first
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = os.getenv('LOG_FILE')
        self.logger = setup_logging(log_level, log_file)
        
        self.logger.info("🚀 Initializing AI Code Review Agent...")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Validate required environment variables
        required_vars = ['ANTHROPIC_API_KEY', 'GITHUB_TOKEN', 'PR_NUMBER', 'REPO_FULL_NAME']
        env_vars = validate_environment_variables(required_vars, self.logger)
        
        # Set instance variables from environment (with config overrides)
        self.anthropic_key = env_vars.get('ANTHROPIC_API_KEY') or self.config.get('anthropic_api_key')
        self.github_token = env_vars.get('GITHUB_TOKEN') or self.config.get('github_token')
        self.pr_number = env_vars.get('PR_NUMBER') or self.config.get('pr_number')
        self.repo = env_vars.get('REPO_FULL_NAME') or self.config.get('repo_full_name')
        
        # Optional variables
        self.jira_url = os.getenv('JIRA_URL') or self.config.get('jira_url')
        self.jira_email = os.getenv('JIRA_EMAIL') or self.config.get('jira_email')
        self.jira_token = os.getenv('JIRA_API_TOKEN') or self.config.get('jira_api_token')
        self.jira_ticket = os.getenv('JIRA_TICKET') or self.config.get('jira_ticket', '')
        self.pr_title = os.getenv('PR_TITLE', '') or self.config.get('pr_title', '')
        self.pr_body = os.getenv('PR_BODY', '') or self.config.get('pr_body', '')
        self.ai_model = os.getenv('AI_MODEL', 'claude-sonnet-4-20250514') or self.config.get('ai_model', 'claude-sonnet-4-20250514')
        
        # Configuration defaults
        self.max_diff_size = int(os.getenv('MAX_DIFF_SIZE', self.config.get('max_diff_size', 100000)))
        self.max_retries = int(os.getenv('MAX_RETRIES', self.config.get('max_retries', 3)))
        
        # Initialize clients
        try:
            self.client = anthropic.Anthropic(api_key=self.anthropic_key)
            self.logger.info("✅ Anthropic client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Anthropic client: {e}")
            raise
        
        # Initialize Jira (with graceful degradation)
        self.jira = self._initialize_jira()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file (YAML or JSON) with environment variable overrides."""
        config = {}
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        try:
                            import yaml
                            config = yaml.safe_load(f) or {}
                        except ImportError:
                            self.logger.warning("PyYAML not installed. Install with: pip install pyyaml")
                            config = {}
                    elif config_file.endswith('.json'):
                        config = json.load(f)
                    else:
                        self.logger.warning(f"Unsupported config file format: {config_file}")
                        config = {}
                    
                    if config:
                        self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return config
    
    def _initialize_jira(self) -> Optional[JIRA]:
        """Initialize Jira client with graceful error handling."""
        if not all([self.jira_url, self.jira_email, self.jira_token]):
            self.logger.info("⚠️  Jira credentials not provided. Skipping Jira integration.")
            return None
        
        try:
            # Validate Jira URL format
            parsed_url = urlparse(self.jira_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                self.logger.warning(f"Invalid Jira URL format: {self.jira_url}")
                return None
            
            self.logger.info(f"🔍 Initializing Jira connection to {parsed_url.netloc}")
            
            jira_client = JIRA(
                server=self.jira_url,
                basic_auth=(self.jira_email, self.jira_token),
                options={'rest_api_version': '3', 'timeout': 30}
            )
            
            # Test connection
            jira_client.current_user()
            self.logger.info("✅ Jira client initialized successfully")
            return jira_client
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to initialize Jira (will continue without it): {e}")
            return None
    
    def read_file(self, filepath: str) -> str:
        """Read file contents with validation and error handling."""
        if not validate_file_path(filepath, self.logger):
            self.logger.error(f"Invalid file path: {filepath}")
            return ""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.debug(f"Read {len(content)} characters from {filepath}")
                return content
        except Exception as e:
            self.logger.error(f"Error reading {filepath}: {e}")
            return ""
    
    @retry_with_backoff(max_retries=3)
    def analyze_code_with_ai(self, diff: str, changed_files: str) -> Dict:
        """Use Claude to analyze the code changes with retry logic."""
        
        # Validate and sanitize inputs
        diff = sanitize_input(diff, max_length=self.max_diff_size * 2)  # Allow some buffer
        changed_files = sanitize_input(changed_files, max_length=10000)
        
        if not validate_diff_size(diff, self.max_diff_size, self.logger):
            raise ValueError(f"Diff size exceeds maximum allowed size of {self.max_diff_size} bytes")
        
        prompt = f"""You are an expert code reviewer. Analyze this pull request and provide a comprehensive review.

PR Title: {self.pr_title}
PR Description: {self.pr_body}

Changed Files:
{changed_files}

Code Diff:
{diff}

Please provide a detailed code review with the following structure:

1. **PR Summary**: A clear, concise description of what this PR does (2-3 sentences)

2. **Code Quality Assessment**: Rate the overall code quality (Excellent/Good/Needs Improvement/Poor) with reasoning

3. **Highlights** (Effective Code Areas): List 3-5 specific things done well in this PR:
   - Well-structured code
   - Good patterns or practices used
   - Performance improvements
   - Security enhancements
   - Good test coverage
   - Clear documentation

4. **Issues & Concerns**: Identify any problems (categorized by severity):
   - 🔴 Critical: Security issues, bugs, breaking changes
   - 🟡 Medium: Code smells, maintainability issues, missing tests
   - 🔵 Minor: Style issues, optimization opportunities, suggestions

5. **Line-Specific Comments**: For each significant issue, provide:
   - File path (relative to repo root, e.g., "src/utils.py")
   - Line number (exact line number in the NEW file version where the issue appears)
   - The concern (brief description of the issue)
   - Suggested fix (actionable suggestion)
   
   Important: Provide exact line numbers from the diff. These will be used to create inline review comments on specific code lines.

6. **Testing Requirements**: Based on the code changes, list what needs to be tested when this PR is merged:
   - Functional testing areas
   - Edge cases to verify
   - Integration points to test
   - Performance considerations
   - Regression risks

7. **Manual Testing Steps**: Provide 5-10 step-by-step manual testing instructions:
   - Setup requirements
   - Test data needed
   - Actions to perform
   - Expected results
   - Things to verify

Format your response as a JSON object with these keys:
- summary
- quality_rating
- quality_reasoning
- highlights (array of strings)
- issues (object with critical, medium, minor arrays)
- line_comments (array of objects with file, line, concern, suggestion)
- testing_requirements (array of strings)
- manual_testing_steps (array of strings)

Be specific, actionable, and focus on real issues. Don't be overly critical - recognize good work."""

        try:
            self.logger.info(f"🤖 Sending code review request to {self.ai_model}...")
            response = self.client.messages.create(
                model=self.ai_model,
                max_tokens=16000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract the JSON from the response
            content = response.content[0].text
            self.logger.debug(f"Received response from AI (length: {len(content)})")
            
            # Try to parse JSON - Claude might wrap it in markdown
            if "```json" in content:
                json_start = content.index("```json") + 7
                json_end = content.rindex("```")
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.index("```") + 3
                json_end = content.rindex("```")
                content = content[json_start:json_end].strip()
            
            review_data = json.loads(content)
            
            # Validate review data structure
            required_keys = ['summary', 'quality_rating', 'highlights', 'issues']
            missing_keys = [key for key in required_keys if key not in review_data]
            if missing_keys:
                self.logger.warning(f"Review data missing keys: {missing_keys}")
            
            # Save the raw review
            with open('review_data.json', 'w') as f:
                json.dump(review_data, f, indent=2)
            
            self.logger.info("✅ AI analysis completed successfully")
            return review_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response as JSON: {e}")
            self.logger.debug(f"Response content: {content[:500]}...")
            raise
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in AI analysis: {e}", exc_info=True)
            raise
    
    def format_pr_comment(self, review: Dict) -> str:
        """Format the review as a GitHub comment."""
        
        comment = f"""## 🤖 AI Code Review

### Summary
{review.get('summary', 'No summary provided')}

### Code Quality: {review.get('quality_rating', 'N/A')}
{review.get('quality_reasoning', '')}

---

### ✨ Highlights (What's Done Well)
"""
        
        for highlight in review.get('highlights', []):
            comment += f"- {highlight}\n"
        
        # Issues section
        issues = review.get('issues', {})
        
        if issues.get('critical'):
            comment += "\n### 🔴 Critical Issues\n"
            for issue in issues['critical']:
                comment += f"- {issue}\n"
        
        if issues.get('medium'):
            comment += "\n### 🟡 Medium Priority\n"
            for issue in issues['medium']:
                comment += f"- {issue}\n"
        
        if issues.get('minor'):
            comment += "\n### 🔵 Minor Suggestions\n"
            for issue in issues['minor']:
                comment += f"- {issue}\n"
        
        # Testing requirements
        comment += "\n---\n\n### 🧪 Testing Requirements\n"
        for req in review.get('testing_requirements', []):
            comment += f"- {req}\n"
        
        # Manual testing steps
        comment += "\n### 📋 Manual Testing Steps\n"
        for i, step in enumerate(review.get('manual_testing_steps', []), 1):
            comment += f"{i}. {step}\n"
        
        comment += "\n---\n*Review generated by AI Code Review Agent using Claude Sonnet 4.5*"
        
        return comment
    
    @retry_with_backoff(max_retries=3)
    def post_github_comment(self, comment: str):
        """Post review comment to GitHub PR with retry logic."""
        url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/comments"
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {'body': comment}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 201:
                self.logger.info("✅ Posted review comment to GitHub PR")
            else:
                self.logger.warning(f"Unexpected status code: {response.status_code}")
        except requests.RequestException as e:
            self.logger.error(f"❌ Failed to post comment to GitHub: {e}")
            if hasattr(e.response, 'text'):
                self.logger.debug(f"Response: {e.response.text}")
            raise
    
    def _calculate_diff_position(self, diff: str, file_path: str, target_line: int) -> Optional[int]:
        """
        Calculate the position in the diff for a given file and line number.
        Position is the line number in the diff counting from the first @@ hunk header.
        
        According to GitHub API, position is the line number in the diff file,
        counting from the first "@@" hunk header. Each hunk resets the position count.
        
        Args:
            diff: The full PR diff
            file_path: Path to the file (relative to repo root)
            target_line: The line number in the new file version
            
        Returns:
            Position in diff (1-based) or None if not found
        """
        try:
            lines = diff.split('\n')
            current_file = None
            in_target_file = False
            hunk_position = 0  # Position within current hunk (from @@ header, 1-based)
            new_file_line = 0  # Current line number in new file
            
            for line in lines:
                # Track file changes
                if line.startswith('diff --git'):
                    # Extract file path from diff header
                    # Format: diff --git a/path/to/file b/path/to/file
                    parts = line.split()
                    if len(parts) >= 4:
                        # Get the 'b' path (new file version)
                        new_file_path = parts[3]
                        if new_file_path.startswith('b/'):
                            current_file = new_file_path[2:]  # Remove 'b/' prefix
                        else:
                            current_file = new_file_path
                        
                        # Check if this is our target file
                        in_target_file = (current_file == file_path)
                        hunk_position = 0
                        new_file_line = 0
                
                # Track hunk headers
                elif line.startswith('@@'):
                    if in_target_file:
                        # Format: @@ -old_start,old_count +new_start,new_count @@
                        # Extract new file line number
                        match = re.search(r'\+(\d+)', line)
                        if match:
                            new_file_line = int(match.group(1)) - 1  # Will be incremented below
                        hunk_position = 0  # Reset hunk position for new hunk
                
                # Count lines in hunk (position is relative to @@ header)
                elif in_target_file:
                    # Only count lines that can receive comments:
                    # - Lines starting with + (additions)
                    # - Lines starting with space (unchanged context lines)
                    # - But NOT lines starting with - (deletions, only in old file)
                    # - And NOT diff metadata lines
                    if line.startswith('+') or (line.startswith(' ') and len(line) > 1 and not line.startswith('---') and not line.startswith('+++')):
                        hunk_position += 1
                        new_file_line += 1
                        
                        # If we've reached the target line, return the position
                        if new_file_line == target_line:
                            return hunk_position
                    elif line.startswith('-'):
                        # Deletion lines don't count toward position or new_file_line
                        # They're only in the old file version
                        pass
                    # Other lines (like ---, +++, index, etc.) are ignored
            
            # If exact line not found, log and return None
            self.logger.debug(f"Line {target_line} not found in diff for {file_path}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error calculating diff position: {e}")
            return None
    
    @retry_with_backoff(max_retries=2)
    def post_line_comments(self, review: Dict):
        """Post line-specific review comments on the PR as inline review comments.
        
        These comments appear directly on the code lines in the PR diff, just like
        when a human reviewer clicks on a line and adds a comment.
        """
        
        line_comments = review.get('line_comments', [])
        if not line_comments:
            self.logger.debug("No line comments to post")
            return
        
        try:
            # Read the diff to calculate positions
            diff = self.read_file('pr_diff.txt')
            if not diff:
                self.logger.warning("⚠️  No diff available for calculating line positions")
                return
            
            # Get the PR details to get the commit SHA
            pr_url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            pr_response = requests.get(pr_url, headers=headers, timeout=30)
            pr_response.raise_for_status()
            
            pr_data = pr_response.json()
            commit_sha = pr_data['head']['sha']
            
            # Post review with comments
            review_url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}/reviews"
            
            comments = []
            for comment in line_comments[:10]:  # Limit to 10 line comments
                # Sanitize file path
                file_path = sanitize_input(comment.get('file', ''), max_length=500)
                if not file_path:
                    continue
                
                # Get line number from comment
                line_number = max(1, int(comment.get('line', 1)))
                
                # Calculate position in diff
                position = self._calculate_diff_position(diff, file_path, line_number)
                
                if position is None:
                    self.logger.debug(f"Could not calculate diff position for {file_path}:{line_number}, skipping")
                    continue
                
                # Format comment body
                concern = comment.get('concern', '')
                suggestion = comment.get('suggestion', '')
                body = f"**{concern}**" if concern else ""
                if suggestion:
                    body += f"\n\n{suggestion}" if body else suggestion
                
                if not body:
                    continue
                
                comments.append({
                    'path': file_path,
                    'position': position,
                    'body': body
                })
            
            if not comments:
                self.logger.warning("No valid line comments to post (could not calculate positions)")
                return
            
            review_data = {
                'commit_id': commit_sha,
                'body': '🤖 AI-identified code review comments',
                'event': 'COMMENT',
                'comments': comments
            }
            
            response = requests.post(review_url, headers=headers, json=review_data, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                self.logger.info(f"✅ Posted {len(comments)} inline line-by-line review comments")
            else:
                self.logger.warning(f"Unexpected status code: {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.warning(f"⚠️  Could not post line comments: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.debug(f"Response: {e.response.text}")
            # Don't raise - line comments are optional
        except Exception as e:
            self.logger.warning(f"⚠️  Error posting line comments: {e}")
            # Don't raise - line comments are optional
    
    def update_jira_ticket(self, review: Dict):
        """Update Jira ticket with testing requirements with graceful error handling."""
        
        if not self.jira or not self.jira_ticket:
            self.logger.info("⚠️  Skipping Jira update (no ticket or credentials)")
            return
        
        try:
            self.logger.info(f"🔍 Attempting to fetch Jira issue: {self.jira_ticket}")
            
            # Validate ticket format
            if not re.match(r'^[A-Z]+-\d+$', self.jira_ticket):
                self.logger.warning(f"Invalid Jira ticket format: {self.jira_ticket}")
                return
            
            # Get the issue
            issue = self.jira.issue(self.jira_ticket)
            self.logger.info(f"✅ Successfully fetched issue: {issue.key}")
            
            # Build ADF content
            content = [
                {
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": [{"type": "text", "text": "Testing Requirements for PR Merge"}]
                },
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": f"From PR #{self.pr_number}: {sanitize_input(self.pr_title, max_length=200)}", "marks": [{"type": "em"}]}
                    ]
                },
                {
                    "type": "heading",
                    "attrs": {"level": 4},
                    "content": [{"type": "text", "text": "What Needs to Be Tested:"}]
                }
            ]
            
            # Add testing requirements as bullet list
            if review.get('testing_requirements'):
                bullet_items = []
                for req in review.get('testing_requirements', []):
                    sanitized_req = sanitize_input(req, max_length=1000)
                    bullet_items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": sanitized_req}]
                        }]
                    })
                content.append({"type": "bulletList", "content": bullet_items})
            
            # Add manual testing steps header
            content.append({
                "type": "heading",
                "attrs": {"level": 4},
                "content": [{"type": "text", "text": "Manual Testing Steps:"}]
            })
            
            # Add manual testing steps as numbered list
            if review.get('manual_testing_steps'):
                ordered_items = []
                for step in review.get('manual_testing_steps', []):
                    sanitized_step = sanitize_input(step, max_length=1000)
                    ordered_items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": sanitized_step}]
                        }]
                    })
                content.append({"type": "orderedList", "content": ordered_items})
            
            # Add separator and link
            content.append({"type": "rule"})
            content.append({
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Auto-generated from ", "marks": [{"type": "em"}]},
                    {
                        "type": "text",
                        "text": f"PR #{self.pr_number}",
                        "marks": [
                            {"type": "em"},
                            {"type": "link", "attrs": {"href": f"https://github.com/{self.repo}/pull/{self.pr_number}"}}
                        ]
                    }
                ]
            })
            
            comment_body = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": content
                }
            }
            
            # Add comment using REST API directly with retry
            url = f"{self.jira_url}/rest/api/3/issue/{self.jira_ticket}/comment"
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.jira_email}:{self.jira_token}'.encode()).decode()}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, headers=headers, json=comment_body, timeout=30)
                response.raise_for_status()
                
                if response.status_code in [200, 201]:
                    self.logger.info(f"✅ Updated Jira ticket {self.jira_ticket}")
                else:
                    self.logger.warning(f"Unexpected status code: {response.status_code}")
            except requests.RequestException as e:
                self.logger.warning(f"⚠️  Failed to update Jira (non-critical): {e}")
                # Don't raise - Jira update is optional
            
        except Exception as e:
            self.logger.warning(f"⚠️  Error updating Jira (non-critical): {e}")
            # Don't raise - Jira update failure shouldn't fail the entire review
    
    def run(self):
        """Main execution flow with comprehensive error handling."""
        try:
            self.logger.info("🚀 Starting AI Code Review Agent...")
            self.logger.info(f"📝 Reviewing PR #{self.pr_number} in {self.repo}")
            
            # Read the diff and changed files
            diff = self.read_file('pr_diff.txt')
            changed_files = self.read_file('changed_files.txt')
            
            if not diff:
                self.logger.warning("⚠️  No diff found, exiting")
                return
            
            self.logger.info(f"📊 Analyzing {len(diff)} characters of code changes...")
            
            # Analyze with AI
            review = self.analyze_code_with_ai(diff, changed_files)
            
            # Format and post PR comment
            comment = self.format_pr_comment(review)
            self.post_github_comment(comment)
            
            # Post line-specific comments
            self.post_line_comments(review)
            
            # Update Jira ticket (non-blocking)
            if self.jira_ticket:
                self.logger.info(f"🎫 Updating Jira ticket: {self.jira_ticket}")
                self.update_jira_ticket(review)
            
            self.logger.info("✅ Code review complete!")
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in code review: {e}", exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    # Allow config file to be specified via environment variable
    config_file = os.getenv('CONFIG_FILE')
    agent = CodeReviewAgent(config_file=config_file)
    agent.run()

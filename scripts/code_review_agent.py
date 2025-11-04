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

class CodeReviewAgent:
    def __init__(self):
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.jira_url = os.getenv('JIRA_URL')
        self.jira_email = os.getenv('JIRA_EMAIL')
        self.jira_token = os.getenv('JIRA_API_TOKEN')
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.pr_number = os.getenv('PR_NUMBER')
        self.repo = os.getenv('REPO_FULL_NAME')
        self.jira_ticket = os.getenv('JIRA_TICKET')
        self.pr_title = os.getenv('PR_TITLE', '')
        self.pr_body = os.getenv('PR_BODY', '')
        
        # Initialize clients
        self.client = anthropic.Anthropic(api_key=self.anthropic_key)
        
        if self.jira_url and self.jira_email and self.jira_token:
            print(f"🔍 Initializing Jira with URL: {self.jira_url}")
            print(f"🔍 Jira Email: {self.jira_email}")
            print(f"🔍 Jira Token exists: {bool(self.jira_token)}")
            print(f"🔍 Jira Token length: {len(self.jira_token) if self.jira_token else 0}")
            
            try:
                self.jira = JIRA(
                    server=self.jira_url,
                    basic_auth=(self.jira_email, self.jira_token),
                    options={'rest_api_version': '3'}
                )
                print("✅ Jira client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Jira: {e}")
                self.jira = None
        else:
            self.jira = None
            print("⚠️  Jira credentials not found. Skipping Jira integration.")
    
    def read_file(self, filepath: str) -> str:
        """Read file contents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    
    def analyze_code_with_ai(self, diff: str, changed_files: str) -> Dict:
        """Use Claude to analyze the code changes"""
        
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
   - File path
   - Approximate line number or section
   - The concern
   - Suggested fix

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
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract the JSON from the response
            content = response.content[0].text
            
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
            
            # Save the raw review
            with open('review_data.json', 'w') as f:
                json.dump(review_data, f, indent=2)
            
            return review_data
            
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            sys.exit(1)
    
    def format_pr_comment(self, review: Dict) -> str:
        """Format the review as a GitHub comment"""
        
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
    
    def post_github_comment(self, comment: str):
        """Post review comment to GitHub PR"""
        url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/comments"
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {'body': comment}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            print("✅ Posted review comment to GitHub PR")
        else:
            print(f"❌ Failed to post comment: {response.status_code}")
            print(response.text)
    
    def post_line_comments(self, review: Dict):
        """Post line-specific review comments on the PR"""
        
        line_comments = review.get('line_comments', [])
        if not line_comments:
            return
        
        # Get the PR details to get the commit SHA
        pr_url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        pr_response = requests.get(pr_url, headers=headers)
        if pr_response.status_code != 200:
            print("Could not fetch PR details for line comments")
            return
        
        pr_data = pr_response.json()
        commit_sha = pr_data['head']['sha']
        
        # Post review with comments
        review_url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}/reviews"
        
        comments = []
        for comment in line_comments[:10]:  # Limit to 10 line comments
            comments.append({
                'path': comment.get('file', ''),
                'body': f"**{comment.get('concern', '')}**\n\n{comment.get('suggestion', '')}",
                'line': comment.get('line', 1)
            })
        
        review_data = {
            'commit_id': commit_sha,
            'body': '🤖 AI-identified code review comments',
            'event': 'COMMENT',
            'comments': comments
        }
        
        response = requests.post(review_url, headers=headers, json=review_data)
        
        if response.status_code == 200:
            print(f"✅ Posted {len(comments)} line-specific comments")
        else:
            print(f"⚠️  Could not post line comments: {response.status_code}")
    
    def update_jira_ticket(self, review: Dict):
        """Update Jira ticket with testing requirements"""
        
        if not self.jira or not self.jira_ticket:
            print("⚠️  Skipping Jira update (no ticket or credentials)")
            return
        
        try:
            print(f"🔍 Attempting to fetch Jira issue: {self.jira_ticket}")
            
            # Get the issue
            issue = self.jira.issue(self.jira_ticket)
            print(f"✅ Successfully fetched issue: {issue.key}")
            
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
                        {"type": "text", "text": f"From PR #{self.pr_number}: {self.pr_title}", "marks": [{"type": "em"}]}
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
                    bullet_items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": req}]
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
                    ordered_items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": step}]
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
            
            # Add comment using REST API directly
            url = f"{self.jira_url}/rest/api/3/issue/{self.jira_ticket}/comment"
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.jira_email}:{self.jira_token}'.encode()).decode()}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=comment_body)
            
            if response.status_code in [200, 201]:
                print(f"✅ Updated Jira ticket {self.jira_ticket}")
            else:
                print(f"❌ Failed to update Jira: {response.status_code}")
                print(response.text)
            
        except Exception as e:
            print(f"❌ Error updating Jira: {e}")
            import traceback
            print(traceback.format_exc())
    
    def run(self):
        """Main execution flow"""
        print("🚀 Starting AI Code Review Agent...")
        print(f"📝 Reviewing PR #{self.pr_number}")
        
        # Read the diff and changed files
        diff = self.read_file('pr_diff.txt')
        changed_files = self.read_file('changed_files.txt')
        
        if not diff:
            print("⚠️  No diff found, exiting")
            return
        
        print(f"📊 Analyzing {len(diff)} characters of code changes...")
        
        # Analyze with AI
        review = self.analyze_code_with_ai(diff, changed_files)
        
        # Format and post PR comment
        comment = self.format_pr_comment(review)
        self.post_github_comment(comment)
        
        # Post line-specific comments
        self.post_line_comments(review)
        
        # Update Jira ticket
        if self.jira_ticket:
            print(f"🎫 Updating Jira ticket: {self.jira_ticket}")
            self.update_jira_ticket(review)
        
        print("✅ Code review complete!")

if __name__ == '__main__':
    agent = CodeReviewAgent()
    agent.run()
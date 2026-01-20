"""Common tool factories for build_debate workflow agents.

Provides reusable tool creation functions to reduce code duplication.
"""

import os
import re
import subprocess
from typing import Dict, Any, List
from langchain_core.tools import tool


def create_read_file_tool(max_length: int = 15000):
    """Create a file reading tool with optional truncation.

    Args:
        max_length: Maximum content length before truncation (default: 15000)

    Returns:
        A LangChain tool for reading files
    """
    @tool
    def read_file(file_path: str) -> str:
        """Read contents of a file.

        Args:
            file_path: Path to the file to read

        Returns:
            File contents as string
        """
        try:
            if not file_path:
                return "ERROR: Empty file path provided"
            if os.path.isdir(file_path):
                return f"ERROR: '{file_path}' is a directory, not a file."
            if not os.path.exists(file_path):
                return f"ERROR: File not found: '{file_path}'"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if max_length and len(content) > max_length:
                content = content[:max_length] + "\n... [TRUNCATED]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    return read_file


def create_write_file_tool():
    """Create a file writing tool with proper directory handling.

    Returns:
        A LangChain tool for writing files
    """
    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file.

        Args:
            file_path: Path to write to
            content: Content to write

        Returns:
            Success message or error
        """
        try:
            if not file_path:
                return "ERROR: Empty file path provided"

            # Normalize path
            file_path = os.path.normpath(file_path)

            # Create parent directory if it exists in the path
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    return write_file


def create_list_directory_tool():
    """Create a directory listing tool.

    Returns:
        A LangChain tool for listing directories
    """
    @tool
    def list_directory(dir_path: str) -> str:
        """List files and subdirectories in a directory.

        Args:
            dir_path: Path to the directory

        Returns:
            Formatted list of directory contents
        """
        try:
            if not dir_path:
                return "ERROR: Empty directory path provided"
            if not os.path.exists(dir_path):
                return f"Directory not found: {dir_path}"
            if not os.path.isdir(dir_path):
                return f"Not a directory: {dir_path}"
            items = []
            for item in sorted(os.listdir(dir_path)):
                full_path = os.path.join(dir_path, item)
                prefix = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
                items.append(f"{prefix} {item}")
            return "\n".join(items) if items else "Empty directory"
        except Exception as e:
            return f"Error listing directory: {e}"

    return list_directory


def create_read_pdf_tool(max_length: int = 50000):
    """Create a PDF reading tool.

    Args:
        max_length: Maximum content length before truncation (default: 50000)

    Returns:
        A LangChain tool for reading PDFs
    """
    @tool
    def read_pdf(pdf_path: str) -> str:
        """Read and extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content from the PDF
        """
        try:
            import fitz  # PyMuPDF

            if not pdf_path:
                return "ERROR: Empty PDF path provided"
            if not os.path.exists(pdf_path):
                return f"PDF file not found: {pdf_path}"

            doc = fitz.open(pdf_path)
            text_content = []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                text_content.append(f"\n--- Page {page_num + 1} ---\n{text}")

            doc.close()

            full_text = "\n".join(text_content)

            # Truncate if too long
            if max_length and len(full_text) > max_length:
                full_text = full_text[:max_length] + "\n\n... [TRUNCATED - PDF too long]"

            return full_text

        except ImportError:
            return "Error: PyMuPDF (fitz) not installed. Run: pip install pymupdf"
        except Exception as e:
            return f"Error reading PDF: {e}"

    return read_pdf


def create_extract_github_urls_tool():
    """Create a tool to extract GitHub URLs from PDF or text.

    Returns:
        A LangChain tool for extracting GitHub URLs
    """
    @tool
    def extract_github_urls(pdf_path: str) -> Dict[str, Any]:
        """Extract GitHub repository URLs from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict with extracted GitHub URLs and metadata
        """
        try:
            import fitz  # PyMuPDF

            if not pdf_path or not os.path.exists(pdf_path):
                return {"success": False, "error": f"PDF not found: {pdf_path}"}

            doc = fitz.open(pdf_path)
            urls = set()

            # GitHub URL patterns
            patterns = [
                r'https?://github\.com/[\w\-]+/[\w\-\.]+',
                r'github\.com/[\w\-]+/[\w\-\.]+',
            ]

            for page in doc:
                # Method 1: Extract from text content
                text = page.get_text()
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        url = match if match.startswith('http') else f'https://{match}'
                        url = re.sub(r'[.,;:)\]]+$', '', url)
                        urls.add(url)

                # Method 2: Extract from hyperlink annotations (IMPORTANT!)
                links = page.get_links()
                for link in links:
                    if 'uri' in link:
                        uri = link['uri']
                        if 'github.com' in uri.lower():
                            urls.add(uri)

            doc.close()

            url_list = sorted(list(urls))

            return {
                "success": True,
                "num_urls": len(url_list),
                "github_urls": url_list
            }

        except ImportError:
            return {"success": False, "error": "PyMuPDF not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return extract_github_urls


def create_clone_github_repo_tool():
    """Create a tool to clone a GitHub repository.

    Returns:
        A LangChain tool for cloning GitHub repos
    """
    import requests

    @tool
    def clone_github_repo(github_url: str, target_dir: str) -> Dict[str, Any]:
        """Clone a GitHub repository to a local directory.

        Args:
            github_url: GitHub repository URL
            target_dir: Directory to clone into

        Returns:
            Dict with clone status and path
        """
        try:
            if not github_url:
                return {"success": False, "error": "Empty GitHub URL"}

            # Normalize URL
            original_url = github_url
            if not github_url.startswith('http'):
                github_url = f'https://{github_url}'
            github_url = github_url.rstrip('/').rstrip('.git')

            # Step 1: Verify repository exists and is accessible
            api_url = github_url.replace("github.com", "api.github.com/repos")
            try:
                resp = requests.get(api_url, timeout=10, headers={"Accept": "application/vnd.github.v3+json"})
                if resp.status_code == 404:
                    return {"success": False, "error": f"Repository not found: {github_url}"}
                elif resp.status_code == 403:
                    return {"success": False, "error": f"Repository access forbidden (may be private): {github_url}"}
                elif resp.status_code != 200:
                    return {"success": False, "error": f"Cannot access repository (HTTP {resp.status_code}): {github_url}"}

                repo_info = resp.json()
                default_branch = repo_info.get("default_branch", "main")
            except requests.RequestException as e:
                return {"success": False, "error": f"Network error checking repository: {e}"}

            os.makedirs(target_dir, exist_ok=True)

            # Extract repo name for subdirectory
            repo_name = github_url.split('/')[-1]
            clone_path = os.path.join(target_dir, repo_name)

            # Step 2: Check if already cloned properly
            if os.path.exists(clone_path):
                # Verify it's a valid git repo with files
                git_dir = os.path.join(clone_path, ".git")
                has_files = any(f for f in os.listdir(clone_path) if f != ".git") if os.path.isdir(clone_path) else False

                if os.path.isdir(git_dir) and has_files:
                    return {
                        "success": True,
                        "path": clone_path,
                        "message": "Repository already cloned"
                    }
                else:
                    # Incomplete clone - remove and retry
                    import shutil
                    shutil.rmtree(clone_path, ignore_errors=True)

            # Step 3: Clone with .git suffix (disable interactive prompts)
            clone_url = f"{github_url}.git"
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"  # Disable git credential prompts
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, clone_path],
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                # Clean up failed clone attempt
                if os.path.exists(clone_path):
                    import shutil
                    shutil.rmtree(clone_path, ignore_errors=True)
                return {"success": False, "error": f"Git clone failed: {error_msg}"}

            # Verify clone succeeded (has files besides .git)
            files = [f for f in os.listdir(clone_path) if f != ".git"]
            if not files:
                return {"success": False, "error": "Clone completed but no files found"}

            return {
                "success": True,
                "path": clone_path,
                "message": f"Cloned to {clone_path}",
                "files_count": len(files)
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Clone timeout (120s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return clone_github_repo


def create_read_github_file_tool():
    """Create a tool to read a file from a cloned GitHub repository.

    Returns:
        A LangChain tool for reading files from cloned repos
    """
    @tool
    def read_github_file(repo_path: str, file_path: str, max_length: int = 30000) -> Dict[str, Any]:
        """Read a specific file from a cloned GitHub repository.

        Args:
            repo_path: Path to the cloned repository
            file_path: Relative path to the file within the repo
            max_length: Maximum content length (default: 30000)

        Returns:
            Dict with file content
        """
        try:
            full_path = os.path.join(repo_path, file_path)

            if not os.path.exists(full_path):
                return {"success": False, "error": f"File not found: {file_path}"}

            if os.path.isdir(full_path):
                # List directory contents
                items = os.listdir(full_path)
                return {
                    "success": True,
                    "is_directory": True,
                    "contents": items
                }

            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) > max_length:
                content = content[:max_length] + "\n... [TRUNCATED]"

            return {
                "success": True,
                "is_directory": False,
                "content": content,
                "path": file_path
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    return read_github_file


def create_list_repo_structure_tool():
    """Create a tool to list the structure of a cloned repository.

    Returns:
        A LangChain tool for listing repo structure
    """
    @tool
    def list_repo_structure(repo_path: str, max_depth: int = 3) -> Dict[str, Any]:
        """List the directory structure of a cloned repository.

        Args:
            repo_path: Path to the cloned repository
            max_depth: Maximum depth to traverse (default: 3)

        Returns:
            Dict with repository structure
        """
        try:
            if not os.path.exists(repo_path):
                return {"success": False, "error": f"Repo not found: {repo_path}"}

            structure = []

            def traverse(path: str, prefix: str = "", depth: int = 0):
                if depth > max_depth:
                    return
                try:
                    items = sorted(os.listdir(path))
                except PermissionError:
                    return

                # Filter out common non-essential directories
                skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}

                for item in items:
                    if item in skip_dirs:
                        continue
                    full_path = os.path.join(path, item)
                    rel_path = os.path.relpath(full_path, repo_path)

                    if os.path.isdir(full_path):
                        structure.append(f"{prefix}{item}/")
                        traverse(full_path, prefix + "  ", depth + 1)
                    else:
                        # Only show Python files and key files
                        if item.endswith(('.py', '.md', '.txt', '.yaml', '.yml', '.json')):
                            structure.append(f"{prefix}{item}")

            traverse(repo_path)

            return {
                "success": True,
                "repo_path": repo_path,
                "structure": "\n".join(structure[:200])  # Limit output
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    return list_repo_structure

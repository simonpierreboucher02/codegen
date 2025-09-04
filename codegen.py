#!/usr/bin/env python3

import os
import sys
import json
import requests
import pathlib
import time
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from functools import wraps
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
from contextlib import contextmanager
import psutil

import click
import yaml
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.tree import Tree
from rich.table import Table
from rich.text import Text
from rich import box
from rich.logging import RichHandler

API_URL = "https://api.openai.com/v1/responses"

# Initialize Rich console
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("codegen")

# Enhanced Configuration models
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    ttl_hours: int = 24
    max_size_mb: int = 100
    directory: str = ".codegen_cache"

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    validate_file_paths: bool = True
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.json', 
        '.yaml', '.yml', '.md', '.txt', '.sh', '.sql', '.go', '.java', '.cpp', '.c'
    ])
    forbidden_paths: List[str] = field(default_factory=lambda: [
        '/etc/', '/var/', '/usr/', '/sys/', '/proc/', 'C:\\Windows\\', 'C:\\System32\\'
    ])

class CodegenConfig(BaseModel):
    """Enhanced configuration for codegen CLI."""
    # Core settings
    model: str = Field(default="gpt-5", description="OpenAI model to use")
    fallback_model: str = Field(default="gpt-4", description="Fallback model if primary fails")
    max_files: int = Field(default=100, description="Maximum files to generate")
    timeout: int = Field(default=120, description="API timeout in seconds")
    
    # UI settings
    show_code_preview: bool = Field(default=True, description="Show code preview in terminal")
    max_preview_lines: int = Field(default=20, description="Max lines to show in code preview")
    theme: str = Field(default="monokai", description="Syntax highlighting theme")
    show_progress: bool = Field(default=True, description="Show progress indicators")
    
    # Advanced settings
    parallel_generation: bool = Field(default=False, description="Generate files in parallel")
    max_workers: int = Field(default=3, description="Max parallel workers")
    smart_retry: bool = Field(default=True, description="Enable intelligent retry logic")
    enable_cache: bool = Field(default=True, description="Enable response caching")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Nested configs
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    security_config: SecurityConfig = Field(default_factory=SecurityConfig)
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        return max(1, min(v, 10))  # Limit between 1 and 10
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return v.upper() if v.upper() in valid_levels else 'INFO'
    
    class Config:
        env_prefix = "CODEGEN_"

def get_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(Panel.fit(
            "[red]❌ OpenAI API Key Required[/red]\n\n"
            "Please set your OpenAI API key:\n"
            "[cyan]export OPENAI_API_KEY='your-api-key-here'[/cyan]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red"
        ))
        sys.exit(1)
    return api_key

def load_config() -> CodegenConfig:
    """Load configuration from YAML file and environment variables."""
    config_data = {}
    
    # Try to load from config.yaml
    script_dir = pathlib.Path(__file__).parent
    config_files = [script_dir / "config.yaml", pathlib.Path.cwd() / "config.yaml"]
    
    for config_file in config_files:
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                    
                # Transform nested configs
                if 'retry' in yaml_config:
                    config_data['retry_config'] = RetryConfig(**yaml_config.pop('retry'))
                if 'cache' in yaml_config:
                    config_data['cache_config'] = CacheConfig(**yaml_config.pop('cache'))
                if 'security' in yaml_config:
                    config_data['security_config'] = SecurityConfig(**yaml_config.pop('security'))
                    
                config_data.update(yaml_config)
                logger.debug(f"Loaded configuration from {config_file}")
                break
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
    
    return CodegenConfig(**config_data)

def create_banner(title: str, subtitle: str = "") -> Panel:
    """Create a beautiful banner for the CLI."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    
    return Panel.fit(
        content,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    )

# Cache Management
class ResponseCache:
    """Intelligent response caching system."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = pathlib.Path(config.directory)
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_cache_key(self, messages: list, model: str) -> str:
        """Generate a cache key from messages and model."""
        content = json.dumps({
            'messages': messages,
            'model': model
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, messages: list, model: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        if not self.config.enabled:
            return None
            
        cache_key = self._get_cache_key(messages, model)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            with self._lock:
                if not cache_file.exists():
                    return None
                
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check expiry
                if datetime.now() - cached_data['timestamp'] > timedelta(hours=self.config.ttl_hours):
                    cache_file.unlink(missing_ok=True)
                    return None
                
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return cached_data['response']
                
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None
    
    def set(self, messages: list, model: str, response: str) -> None:
        """Cache a response."""
        if not self.config.enabled:
            return
            
        cache_key = self._get_cache_key(messages, model)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            with self._lock:
                cached_data = {
                    'response': response,
                    'timestamp': datetime.now(),
                    'model': model
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                    
                logger.debug(f"Cached response for key {cache_key[:8]}...")
                
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    def cleanup(self) -> int:
        """Remove expired cache entries and return count removed."""
        if not self.config.enabled:
            return 0
            
        removed_count = 0
        try:
            with self._lock:
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        if datetime.now() - cached_data['timestamp'] > timedelta(hours=self.config.ttl_hours):
                            cache_file.unlink()
                            removed_count += 1
                            
                    except Exception:
                        cache_file.unlink(missing_ok=True)
                        removed_count += 1
                        
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")
            
        return removed_count

# Smart Retry System
def smart_retry(config: RetryConfig):
    """Decorator for intelligent retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    if e.response.status_code not in config.retry_on_status:
                        raise  # Don't retry on client errors like 401, 403
                        
                except (requests.exceptions.Timeout, 
                       requests.exceptions.ConnectionError,
                       requests.exceptions.RequestException) as e:
                    last_exception = e
                    
                except Exception as e:
                    # Don't retry on non-network exceptions
                    raise
                
                if attempt < config.max_attempts - 1:
                    delay = min(
                        config.base_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    logger.warning(f"Attempt {attempt + 1}/{config.max_attempts} failed, retrying in {delay:.1f}s: {last_exception}")
                    time.sleep(delay)
                    
            raise last_exception
        return wrapper
    return decorator

# Performance Monitoring
@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_response_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        total_cache_attempts = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_cache_attempts * 100) if total_cache_attempts > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / self.successful_requests if self.successful_requests > 0 else 0

# Global metrics instance
metrics = PerformanceMetrics()

# Health Check System
class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, config: CodegenConfig):
        self.config = config
        self.start_time = datetime.now()
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API connectivity and response time."""
        try:
            start = time.time()
            # Simple health check with minimal payload
            test_messages = [{"role": "user", "content": "health check"}]
            
            api_key = get_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "input": test_messages,
                "text": {"format": {"type": "text"}, "verbosity": "minimal"},
                "reasoning": {"effort": "minimal"},
                "tools": [],
                "store": False
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            response_time = time.time() - start
            
            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "response_time_ms": round(response_time * 1000, 2),
                "status_code": response.status_code,
                "model": self.config.model
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        import psutil
        
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                "memory_usage_percent": memory.percent,
                "memory_available_mb": round(memory.available / (1024**2), 1),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "cpu_count": psutil.cpu_count()
            }
        except ImportError:
            return {"status": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime.total_seconds()),
            "version": "2.0.0",  # Updated version
            "config": {
                "model": self.config.model,
                "fallback_model": self.config.fallback_model,
                "cache_enabled": self.config.enable_cache,
                "parallel_enabled": self.config.parallel_generation
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate_percent": round(metrics.success_rate, 1),
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "cache_hit_rate_percent": round(metrics.cache_hit_rate, 1),
                "avg_response_time_seconds": round(metrics.avg_response_time, 2)
            }
        }
        
        # Add API health check
        status["api_health"] = self.check_api_health()
        
        # Add system resources if available
        status["system_resources"] = self.check_system_resources()
        
        # Add cache status
        if cache and self.config.enable_cache:
            cache_dir = pathlib.Path(self.config.cache_config.directory)
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.cache"))
                cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
                status["cache_status"] = {
                    "cache_files": len(cache_files),
                    "cache_size_mb": round(cache_size_mb, 1),
                    "max_size_mb": self.config.cache_config.max_size_mb
                }
        
        return status

# Enhanced Monitoring
def show_performance_summary(config: CodegenConfig, start_time: float, files_generated: int):
    """Show comprehensive performance summary."""
    elapsed_time = time.time() - start_time
    
    # Create metrics table
    metrics_table = Table(title="📊 Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")
    metrics_table.add_column("Details", style="dim")
    
    metrics_table.add_row(
        "Success Rate", 
        f"{metrics.success_rate:.1f}%",
        f"{metrics.successful_requests}/{metrics.total_requests} requests"
    )
    
    metrics_table.add_row(
        "Cache Performance",
        f"{metrics.cache_hit_rate:.1f}%",
        f"{metrics.cache_hits} hits, {metrics.cache_misses} misses"
    )
    
    metrics_table.add_row(
        "Average Response Time",
        f"{metrics.avg_response_time:.2f}s",
        f"Total: {metrics.total_response_time:.1f}s"
    )
    
    metrics_table.add_row(
        "Generation Speed",
        f"{files_generated/elapsed_time:.1f} files/min",
        f"{files_generated} files in {elapsed_time:.1f}s"
    )
    
    if config.enable_cache and cache:
        cache_dir = pathlib.Path(config.cache_config.directory)
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.cache"))
            cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
            metrics_table.add_row(
                "Cache Size",
                f"{cache_size_mb:.1f} MB",
                f"{len(cache_files)} cached responses"
            )
    
    console.print(metrics_table)
    console.print()

# Progress Enhancement
@contextmanager
def enhanced_progress(total: int, description: str = "Processing"):
    """Enhanced progress context manager with metrics."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.completed}[/green]/[cyan]{task.total}[/cyan] files"),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task(description, total=total)
        
        def update_progress(advance: int = 1, description: str = None):
            progress.advance(task, advance)
            if description:
                progress.update(task, description=description)
        
        yield update_progress

# Initialize cache
cache = None

def call_openai_api(messages: list, config: CodegenConfig, use_fallback: bool = False) -> str:
    """Enhanced API call with caching, retry, and fallback."""
    global cache, metrics
    
    # Initialize cache if needed
    if cache is None and config.enable_cache:
        cache = ResponseCache(config.cache_config)
        cache.cleanup()  # Clean expired entries
    
    model_to_use = config.fallback_model if use_fallback else config.model
    
    # Try cache first
    if cache:
        cached_response = cache.get(messages, model_to_use)
        if cached_response:
            metrics.cache_hits += 1
            metrics.successful_requests += 1
            metrics.total_requests += 1
            return cached_response
        metrics.cache_misses += 1
    
    # Prepare API call
    @smart_retry(config.retry_config)
    def _make_api_call():
        api_key = get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_to_use,
            "input": messages,
            "text": {
                "format": {
                    "type": "text"
                },
                "verbosity": "medium"
            },
            "reasoning": {
                "effort": "minimal"
            },
            "tools": [],
            "store": True,
            "include": [
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ]
        }
        
        start_time = time.time()
        
        with console.status(f"[bold green]Calling {model_to_use}...", spinner="dots"):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=config.timeout)
            response.raise_for_status()
        
        # Update metrics
        response_time = time.time() - start_time
        metrics.total_response_time += response_time
        metrics.total_requests += 1
        
        return response.json()
    
    try:
        result = _make_api_call()
        
        # Parse response
        try:
            if "output" in result and len(result["output"]) > 1:
                message_output = result["output"][1]
                if "content" in message_output and len(message_output["content"]) > 0:
                    content = message_output["content"][0]["text"].strip()
                    
                    # Cache successful response
                    if cache:
                        cache.set(messages, model_to_use, content)
                    
                    metrics.successful_requests += 1
                    return content
            
            raise Exception("Unexpected response structure")
            
        except (KeyError, IndexError, TypeError) as e:
            console.print(f"[dim]Debug - API Response keys: {list(result.keys())}[/dim]")
            console.print(f"[dim]Debug - Output structure: {result.get('output', 'No output key')}[/dim]")
            raise Exception(f"Failed to parse {model_to_use} response: {e}")
    
    except Exception as e:
        metrics.failed_requests += 1
        
        # Try fallback model if primary fails
        if not use_fallback and config.fallback_model != config.model:
            logger.warning(f"Primary model {config.model} failed, trying fallback {config.fallback_model}")
            try:
                return call_openai_api(messages, config, use_fallback=True)
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
        
        # Final failure handling
        if isinstance(e, requests.exceptions.RequestException):
            console.print(f"[red]❌ API request failed: {e}[/red]")
            logger.error(f"API request failed: {e}")
        else:
            console.print(f"[red]❌ API error: {e}[/red]")
            logger.error(f"API error: {e}")
        
        sys.exit(1)

def load_system_prompt() -> str:
    """Load the system prompt from file."""
    script_dir = pathlib.Path(__file__).parent
    prompt_file = script_dir / "system_prompt.txt"
    
    if not prompt_file.exists():
        console.print(f"[red]❌ Error: system_prompt.txt not found at {prompt_file}[/red]")
        sys.exit(1)
        
    try:
        return prompt_file.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]❌ Error reading system prompt: {e}[/red]")
        sys.exit(1)

def get_language_from_extension(file_path: str) -> str:
    """Get programming language from file extension."""
    extension_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.xml': 'xml',
        '.md': 'markdown',
        '.sql': 'sql',
        '.sh': 'bash',
        '.rs': 'rust',
        '.go': 'go',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.dart': 'dart',
        '.vue': 'vue',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini'
    }
    
    ext = pathlib.Path(file_path).suffix.lower()
    return extension_map.get(ext, 'text')

def show_code_preview(file_data: Dict[str, Any], config: CodegenConfig) -> None:
    """Show a beautiful preview of the generated code."""
    if not config.show_code_preview:
        return
        
    file_path = file_data.get("path", "")
    content = file_data.get("content", "")
    language = file_data.get("language", get_language_from_extension(file_path)).lower()
    
    if not content.strip():
        return
    
    # Limit preview length
    lines = content.split('\n')
    if len(lines) > config.max_preview_lines:
        preview_content = '\n'.join(lines[:config.max_preview_lines])
        preview_content += f'\n\n... ({len(lines) - config.max_preview_lines} more lines)'
    else:
        preview_content = content
    
    # Create syntax highlighted code
    try:
        syntax = Syntax(
            preview_content, 
            language, 
            theme="monokai", 
            line_numbers=True,
            background_color="default"
        )
        
        console.print(Panel(
            syntax,
            title=f"[bold cyan]📄 {pathlib.Path(file_path).name}[/bold cyan] [dim]({language})[/dim]",
            border_style="blue",
            expand=False
        ))
        console.print()  # Add spacing
    except Exception as e:
        # Fallback to plain text if syntax highlighting fails
        console.print(Panel(
            Text(preview_content, style="dim"),
            title=f"[bold cyan]📄 {pathlib.Path(file_path).name}[/bold cyan]",
            border_style="blue",
            expand=False
        ))
        console.print()

def validate_file_security(file_json: Dict[str, Any], config: CodegenConfig) -> bool:
    """Validate file for security concerns."""
    if not config.security_config.validate_file_paths:
        return True
        
    file_path = file_json.get("path", "")
    content = file_json.get("content", "")
    
    # Check file path safety
    for forbidden in config.security_config.forbidden_paths:
        if forbidden.lower() in file_path.lower():
            logger.warning(f"Blocked potentially unsafe path: {file_path}")
            return False
    
    # Check file extension
    ext = pathlib.Path(file_path).suffix.lower()
    if ext and ext not in config.security_config.allowed_extensions:
        logger.warning(f"Blocked unsupported file extension: {ext}")
        return False
    
    # Check file size
    content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
    if content_size_mb > config.security_config.max_file_size_mb:
        logger.warning(f"Blocked oversized file: {content_size_mb:.1f}MB > {config.security_config.max_file_size_mb}MB")
        return False
    
    # Check for suspicious content patterns
    suspicious_patterns = [
        'rm -rf /', 'format c:', 'del /f /q', '__import__("os")',
        'eval(', 'exec(', 'subprocess.', 'os.system('
    ]
    
    content_lower = content.lower()
    for pattern in suspicious_patterns:
        if pattern in content_lower:
            logger.warning(f"Blocked file with suspicious content pattern: {pattern}")
            return False
    
    return True

def save_file(file_json: Dict[str, Any], root_path: pathlib.Path, config: CodegenConfig) -> bool:
    """Enhanced file saving with security validation."""
    try:
        # Security validation
        if not validate_file_security(file_json, config):
            console.print(f"[red]🛡️  Security check failed for {file_json.get('path', 'unknown')}[/red]")
            return False
        
        # Handle both absolute paths and relative paths from JSON
        file_path_str = file_json["path"]
        if file_path_str.startswith("/"):
            file_path_str = file_path_str[1:]  # Remove leading slash for relative path
            
        full_path = root_path / file_path_str
        
        # Additional path traversal protection
        try:
            full_path.resolve().relative_to(root_path.resolve())
        except ValueError:
            logger.warning(f"Blocked path traversal attempt: {file_path_str}")
            return False
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file content with atomic operation
        content = file_json.get("content", "")
        temp_path = full_path.with_suffix(full_path.suffix + '.tmp')
        
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            temp_path.replace(full_path)  # Atomic rename
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        
        # Show success message with file info
        file_size = len(content.encode('utf-8'))
        lines_count = content.count('\n') + 1 if content else 0
        
        console.print(f"[green]✓[/green] [bold]{full_path.name}[/bold] [dim]({file_size:,} bytes, {lines_count} lines)[/dim]")
        
        # Show code preview
        show_code_preview(file_json, config)
        
        logger.info(f"Saved file: {full_path}")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error saving {file_json.get('path', 'unknown')}: {e}[/red]")
        logger.error(f"Error saving file {file_json.get('path', 'unknown')}: {e}")
        return False

def parse_project_tree(tree_json: Dict[str, Any]) -> str:
    """Extract project name from tree JSON."""
    try:
        return tree_json.get("name", "generated-project")
    except Exception as e:
        console.print(f"[red]❌ Error parsing project tree: {e}[/red]")
        logger.error(f"Error parsing project tree: {e}")
        return "generated-project"

def display_project_tree(tree_data: Dict[str, Any]) -> None:
    """Display the project tree in a beautiful format."""
    def add_to_tree(tree_node: Tree, item: Dict[str, Any]) -> None:
        name = item.get("name", "unknown")
        item_type = item.get("type", "unknown")
        
        if item_type == "directory":
            icon = "📁"
            style = "bold blue"
        else:
            icon = "📄"
            style = "green"
            
        node = tree_node.add(f"{icon} [style]{name}[/style]".replace("style", style))
        
        # Add children recursively
        children = item.get("children", [])
        for child in children:
            add_to_tree(node, child)
    
    tree = Tree(
        f"🌳 [bold cyan]{tree_data.get('name', 'Project')}[/bold cyan]",
        guide_style="bright_blue"
    )
    
    for child in tree_data.get("children", []):
        add_to_tree(tree, child)
    
    console.print(Panel(
        tree,
        title="[bold green]Project Structure[/bold green]",
        border_style="green",
        expand=False
    ))
    console.print()

def count_files_in_tree(tree_data: Dict[str, Any]) -> int:
    """Count total files in the project tree."""
    def count_files(item: Dict[str, Any]) -> int:
        if item.get("type") == "file":
            return 1
        
        count = 0
        for child in item.get("children", []):
            count += count_files(child)
        return count
    
    return count_files(tree_data)

# Initialize global health checker
health_checker = None

@click.group()
@click.version_option(version="2.0.0", prog_name="codegen")
def cli():
    """🚀 Enhanced AI Code Generator - Production Ready"""
    pass

@cli.command()
@click.argument('description', required=False)
@click.option('--model', default='gpt-5', help='OpenAI model to use')
@click.option('--fallback-model', default='gpt-4', help='Fallback model if primary fails')
@click.option('--max-files', default=100, help='Maximum files to generate')
@click.option('--no-preview', is_flag=True, help='Disable code preview')
@click.option('--max-preview-lines', default=20, help='Max lines in code preview')
@click.option('--no-cache', is_flag=True, help='Disable response caching')
@click.option('--parallel', is_flag=True, help='Enable parallel file generation')
@click.option('--max-workers', default=3, help='Max parallel workers')
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
@click.option('--template', help='Use a predefined template')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', help='Path to config file')
def generate(
    description: Optional[str],
    model: str,
    fallback_model: str,
    max_files: int,
    no_preview: bool,
    max_preview_lines: int,
    no_cache: bool,
    parallel: bool,
    max_workers: int,
    interactive: bool,
    template: Optional[str],
    verbose: bool,
    config: Optional[str]
):
    """🚀 Generate complete projects from simple descriptions using AI.
    
    DESCRIPTION: Describe the project you want to generate (optional if using interactive mode)
    """
    global health_checker
    
    # Handle interactive mode or template selection
    if interactive or template or not description:
        from templates import InteractiveMode, get_template_description, TEMPLATES
        
        if template:
            if template not in TEMPLATES:
                console.print(f"[red]❌ Template '{template}' not found[/red]")
                console.print(f"[dim]Available templates: {', '.join(TEMPLATES.keys())}[/dim]")
                sys.exit(1)
            description = get_template_description(template)
        elif interactive or not description:
            interactive_mode = InteractiveMode()
            description = interactive_mode.run()
        
        if not description:
            console.print("[red]❌ No project description provided[/red]")
            sys.exit(1)
    # Load configuration with command-line overrides
    config_data = {
        'model': model,
        'fallback_model': fallback_model,
        'max_files': max_files,
        'show_code_preview': not no_preview,
        'max_preview_lines': max_preview_lines,
        'enable_cache': not no_cache,
        'parallel_generation': parallel,
        'max_workers': max_workers,
        'log_level': "DEBUG" if verbose else "INFO"
    }
    
    # Load base config from file
    base_config = load_config()
    
    # Override with command line arguments
    for key, value in config_data.items():
        setattr(base_config, key, value)
    
    config = base_config
    
    # Initialize health checker
    health_checker = HealthChecker(config)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show enhanced banner
    subtitle_parts = [f"Powered by {config.model}"]
    if config.fallback_model != config.model:
        subtitle_parts.append(f"Fallback: {config.fallback_model}")
    if config.enable_cache:
        subtitle_parts.append("Cached")
    if config.parallel_generation:
        subtitle_parts.append(f"Parallel ({config.max_workers} workers)")
    
    console.print(create_banner(
        "🚀 CODEGEN v2.0 - Enhanced AI Code Generator", 
        " • ".join(subtitle_parts)
    ))
    console.print()
    
    system_prompt = load_system_prompt()
    
    # Initialize conversation history
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": description
        }
    ]

    # Show project info
    info_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    info_table.add_row("[cyan]Description[/cyan]", f"[white]{description}[/white]")
    info_table.add_row("[cyan]Model[/cyan]", f"[yellow]{config.model}[/yellow]")
    info_table.add_row("[cyan]Fallback Model[/cyan]", f"[yellow]{config.fallback_model}[/yellow]")
    info_table.add_row("[cyan]Max Files[/cyan]", f"[magenta]{config.max_files}[/magenta]")
    info_table.add_row("[cyan]Cache Enabled[/cyan]", f"[green]{'Yes' if config.enable_cache else 'No'}[/green]")
    info_table.add_row("[cyan]Parallel Mode[/cyan]", f"[green]{'Yes' if config.parallel_generation else 'No'}[/green]")
    info_table.add_row("[cyan]Timestamp[/cyan]", f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    console.print(Panel(
        info_table,
        title="[bold blue]📄 Generation Details[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    # Phase 1: Get project tree
    console.print("[bold cyan]🌳 Phase 1: Generating Project Structure[/bold cyan]")
    console.print()
    
    try:
        tree_response = call_openai_api(messages, config)
        tree_data = json.loads(tree_response)
        project_name = parse_project_tree(tree_data)
        
        # Display the tree
        display_project_tree(tree_data)
        
        # Count files
        total_files = count_files_in_tree(tree_data)
        console.print(f"[bold green]✓[/bold green] Project structure created with [bold]{total_files}[/bold] files to generate")
        console.print()
        
        # Create project directory
        project_path = pathlib.Path(project_name)
        project_path.mkdir(exist_ok=True)
        
        # Save tree to project.json
        tree_file = project_path / "project.json"
        with open(tree_file, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
            
        console.print(f"[dim]💾 Project tree saved to {tree_file}[/dim]")
        console.print()
        
        # Add the tree response to conversation history
        messages.append({
            "role": "assistant",
            "content": tree_response
        })
        
    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Failed to parse project tree JSON: {e}[/red]")
        console.print(f"[dim]Raw response: {tree_response[:200] + '...' if len(tree_response) > 200 else tree_response}[/dim]")
        logger.error(f"JSON decode error: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error in Phase 1: {e}[/red]")
        logger.error(f"Phase 1 error: {e}")
        sys.exit(1)

    # Phase 2: Generate files in sequence
    console.print("[bold cyan]📝 Phase 2: Generating Files[/bold cyan]")
    console.print()
    
    file_count = 0
    success_count = 0
    start_time = time.time()
    
    # Enhanced progress tracking
    with enhanced_progress(min(total_files, config.max_files), "[green]Generating files...") as update_progress:
        
        while file_count < config.max_files:
            try:
                # Request next file
                messages.append({
                    "role": "user",
                    "content": "continue"
                })
                
                response = call_openai_api(messages, config).strip()
                
                # Check if generation is complete
                if response == "generation complete":
                    progress.update(task, description="[green]Generation complete! ✨")
                    break
                    
                # Try to parse as file JSON
                try:
                    file_data = json.loads(response)
                    
                    # Validate it's a file object
                    if file_data.get("type") != "file":
                        console.print(f"[yellow]⚠️  Skipping non-file object: {file_data.get('name', 'unknown')}[/yellow]")
                        continue
                        
                    # Save the file
                    if save_file(file_data, project_path, config):
                        success_count += 1
                        
                    file_count += 1
                    update_progress(1, f"[green]Generated {success_count}/{file_count} files")
                    
                    # Add response to conversation history
                    messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except json.JSONDecodeError as e:
                    console.print(f"[red]❌ Failed to parse file JSON: {e}[/red]")
                    console.print(f"[dim]Raw response: {response[:100] + '...' if len(response) > 100 else response}[/dim]")
                    logger.warning(f"JSON decode error: {e}")
                    continue
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]⏹️  Generation stopped by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error generating file: {e}[/red]")
                logger.error(f"File generation error: {e}")
                continue
        
        # Final update
        update_progress(0, f"[green]Completed: {success_count} files generated")
    
    console.print()
    
    # Generation complete summary
    elapsed_time = time.time() - start_time
    
    if file_count >= config.max_files:
        console.print(f"[yellow]⚠️  Reached maximum file limit ({config.max_files})[/yellow]")
    
    # Success summary
    summary_panel = Panel.fit(
        f"[bold green]Project Generated Successfully! 🎉[/bold green]\n\n"
        f"[cyan]Project Name:[/cyan] [bold]{project_name}[/bold]\n"
        f"[cyan]Files Generated:[/cyan] [bold green]{success_count}[/bold green] / {file_count} attempted\n"
        f"[cyan]Time Elapsed:[/cyan] [bold]{elapsed_time:.1f}s[/bold]\n"
        f"[cyan]Location:[/cyan] [bold blue]{project_path.absolute()}[/bold blue]",
        title="[bold green]✓ Generation Complete[/bold green]",
        border_style="green"
    )
    
    console.print(summary_panel)
    console.print()
    
    # Show performance metrics
    show_performance_summary(config, start_time, success_count)
    
    # Log summary
    logger.info(f"Project '{project_name}' generated with {success_count} files in {elapsed_time:.1f}s")
    
    # Cleanup cache if needed
    if cache and config.enable_cache:
        removed = cache.cleanup()
        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired cache entries")

@cli.command()
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
def status(format: str):
    """Show system status and health metrics."""
    if not health_checker:
        # Create a basic health checker
        config = load_config()
        checker = HealthChecker(config)
    else:
        checker = health_checker
    
    status_data = checker.get_comprehensive_status()
    
    if format == 'json':
        console.print(json.dumps(status_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(status_data, default_flow_style=False))
    else:
        # Table format
        table = Table(title="🏥 System Health Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # API Health
        api_health = status_data.get('api_health', {})
        api_status = api_health.get('status', 'unknown')
        api_color = 'green' if api_status == 'healthy' else 'red'
        table.add_row(
            "API Connectivity",
            f"[{api_color}]{api_status.title()}[/{api_color}]",
            f"Response time: {api_health.get('response_time_ms', 'N/A')}ms"
        )
        
        # Metrics
        metrics_data = status_data.get('metrics', {})
        success_rate = metrics_data.get('success_rate_percent', 0)
        success_color = 'green' if success_rate >= 95 else 'yellow' if success_rate >= 80 else 'red'
        table.add_row(
            "Request Success Rate",
            f"[{success_color}]{success_rate}%[/{success_color}]",
            f"{metrics_data.get('successful_requests', 0)}/{metrics_data.get('total_requests', 0)} requests"
        )
        
        # Cache
        cache_hit_rate = metrics_data.get('cache_hit_rate_percent', 0)
        cache_color = 'green' if cache_hit_rate >= 70 else 'yellow' if cache_hit_rate >= 40 else 'red'
        table.add_row(
            "Cache Performance",
            f"[{cache_color}]{cache_hit_rate}%[/{cache_color}] hit rate",
            f"{metrics_data.get('cache_hits', 0)} hits, {metrics_data.get('cache_misses', 0)} misses"
        )
        
        # System Resources (if available)
        resources = status_data.get('system_resources', {})
        if 'memory_usage_percent' in resources:
            memory_usage = resources['memory_usage_percent']
            memory_color = 'red' if memory_usage >= 90 else 'yellow' if memory_usage >= 75 else 'green'
            table.add_row(
                "Memory Usage",
                f"[{memory_color}]{memory_usage}%[/{memory_color}]",
                f"{resources.get('memory_available_mb', 'N/A')} MB available"
            )
        
        console.print(table)
        
        # Uptime info
        uptime_seconds = status_data.get('uptime_seconds', 0)
        uptime_str = str(timedelta(seconds=uptime_seconds))
        console.print(f"\n[dim]Uptime: {uptime_str}[/dim]")
        console.print(f"[dim]Version: {status_data.get('version', 'Unknown')}[/dim]")

@cli.command()
def templates():
    """List available project templates."""
    from templates import TEMPLATES
    
    table = Table(title="📋 Available Project Templates", box=box.ROUNDED)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Usage", style="yellow")
    
    for key, template in TEMPLATES.items():
        table.add_row(
            template.name,
            template.category,
            template.description,
            f"--template {key}"
        )
    
    console.print(table)

@cli.command()
@click.option('--expired-only', is_flag=True, help='Clean only expired entries')
def clean_cache(expired_only: bool):
    """Clean the response cache."""
    config = load_config()
    
    if not config.enable_cache:
        console.print("[yellow]⚠️  Cache is disabled in configuration[/yellow]")
        return
    
    cache_instance = ResponseCache(config.cache_config)
    
    if expired_only:
        removed = cache_instance.cleanup()
        console.print(f"[green]✓[/green] Removed {removed} expired cache entries")
    else:
        # Remove all cache files
        cache_dir = pathlib.Path(config.cache_config.directory)
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.cache"))
            for cache_file in cache_files:
                cache_file.unlink()
            console.print(f"[green]✓[/green] Removed {len(cache_files)} cache entries")
        else:
            console.print("[dim]No cache directory found[/dim]")

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️  Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
#!/usr/bin/env python3
"""
Template system for codegen CLI
Provides pre-built project templates and interactive mode
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

console = Console()

@dataclass
class ProjectTemplate:
    """Represents a project template."""
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    prompt_template: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    
    def render_prompt(self, variables: Dict[str, str] = None) -> str:
        """Render the template with provided variables."""
        if variables is None:
            variables = {}
        
        # Merge with default variables
        all_vars = {**self.variables, **variables}
        
        prompt = self.prompt_template
        for key, value in all_vars.items():
            prompt = prompt.replace(f"{{{key}}}", value)
            
        return prompt

# Built-in templates
TEMPLATES = {
    "webapp": ProjectTemplate(
        name="Modern Web Application",
        description="Full-stack web application with React frontend and Node.js backend",
        category="Web Development",
        tags=["react", "nodejs", "typescript", "mongodb"],
        prompt_template="""Create a modern full-stack web application called '{project_name}' with the following specifications:

Frontend:
- React 18 with TypeScript
- Tailwind CSS for styling
- React Router for navigation
- Axios for API calls
- Modern component architecture

Backend:
- Node.js with Express
- TypeScript configuration
- MongoDB integration with Mongoose
- JWT authentication
- RESTful API design
- Input validation with Joi

Features to implement:
{features}

Additional requirements:
- Responsive design
- Error handling and loading states  
- Environment configuration
- Docker setup
- Basic testing structure
- README with setup instructions

Make it production-ready with proper folder structure, TypeScript types, and modern best practices.""",
        variables={
            "project_name": "my-webapp",
            "features": "- User authentication and registration\n- Dashboard with data visualization\n- CRUD operations for main entities\n- Search and filtering capabilities"
        }
    ),
    
    "api": ProjectTemplate(
        name="REST API Service",
        description="Scalable REST API with database integration",
        category="Backend",
        tags=["api", "nodejs", "express", "database"],
        prompt_template="""Create a professional REST API service called '{project_name}' with these specifications:

Technology Stack:
- Node.js with Express and TypeScript
- {database} database with proper ORM/ODM
- JWT authentication and authorization
- Input validation and sanitization
- API documentation with Swagger/OpenAPI

Core Features:
{features}

API Requirements:
- RESTful endpoints with proper HTTP methods
- Pagination, filtering, and sorting
- Error handling with consistent response format
- Rate limiting and security middleware
- CORS configuration
- Request logging and monitoring

Infrastructure:
- Docker containerization
- Environment-based configuration
- Database migrations/seeds
- Health check endpoints
- Comprehensive error handling
- API versioning strategy

Testing:
- Unit tests with Jest
- Integration tests for API endpoints
- Test database configuration

Make it enterprise-ready with proper validation, security, and documentation.""",
        variables={
            "project_name": "my-api",
            "database": "PostgreSQL",
            "features": "- User management and authentication\n- Resource CRUD operations\n- File upload handling\n- Email notifications\n- Admin dashboard API"
        }
    ),
    
    "mobile": ProjectTemplate(
        name="React Native Mobile App",
        description="Cross-platform mobile application",
        category="Mobile",
        tags=["react-native", "mobile", "typescript"],
        prompt_template="""Create a React Native mobile application called '{project_name}' with these specifications:

Technology Stack:
- React Native with TypeScript
- React Navigation for screen management
- React Native Paper or NativeBase for UI components
- Async Storage for local data persistence
- React Query for API state management

Core Features:
{features}

Mobile-Specific Requirements:
- Responsive design for different screen sizes
- Platform-specific code where necessary (iOS/Android)
- Push notifications setup
- Offline capability with data synchronization
- Biometric authentication support
- Camera and photo library integration
- GPS location services

Architecture:
- Component-based architecture with hooks
- Global state management (Context API or Redux Toolkit)
- API integration with proper error handling
- Form validation and user input handling

Development Setup:
- Metro bundler configuration
- Development and production build configs
- Flipper integration for debugging
- Automated testing setup with Jest and Testing Library

Make it production-ready with proper navigation, state management, and mobile UX patterns.""",
        variables={
            "project_name": "MyMobileApp",
            "features": "- User authentication and profile management\n- Real-time chat functionality\n- Photo sharing with filters\n- Social feed with likes and comments\n- Push notifications for interactions"
        }
    ),
    
    "python": ProjectTemplate(
        name="Python CLI Application",
        description="Professional command-line tool with packaging",
        category="CLI Tools",
        tags=["python", "cli", "packaging"],
        prompt_template="""Create a professional Python CLI application called '{project_name}' with these specifications:

Technology Stack:
- Python 3.11+ with type hints
- Click for command-line interface
- Rich for beautiful terminal output
- Pydantic for data validation
- Requests for HTTP operations

Core Features:
{features}

CLI Requirements:
- Multiple subcommands with options and arguments
- Interactive prompts with validation
- Progress bars and status indicators
- Colored output with formatting
- Configuration file support (YAML/JSON)
- Comprehensive help documentation

Code Quality:
- Type annotations throughout
- Docstrings with Sphinx format
- Error handling with custom exceptions
- Logging with different levels
- Input validation and sanitization

Development Setup:
- Poetry for dependency management
- Pre-commit hooks with black, flake8, mypy
- pytest for testing with fixtures
- GitHub Actions for CI/CD
- Makefile for common tasks

Packaging:
- Setup for PyPI distribution
- Console scripts entry points
- Version management
- README with usage examples

Make it a production-ready CLI tool with professional packaging and documentation.""",
        variables={
            "project_name": "my-cli-tool",
            "features": "- File processing and transformation\n- API integration with authentication\n- Data export in multiple formats\n- Batch operations with progress tracking\n- Configuration management"
        }
    ),
    
    "data": ProjectTemplate(
        name="Data Analysis Pipeline",
        description="Data processing and analysis toolkit",
        category="Data Science",
        tags=["python", "pandas", "jupyter", "data"],
        prompt_template="""Create a comprehensive data analysis pipeline called '{project_name}' with these specifications:

Technology Stack:
- Python 3.11+ with scientific computing libraries
- Pandas for data manipulation
- NumPy for numerical operations
- Matplotlib/Plotly for visualization
- Jupyter notebooks for exploration
- Apache Airflow for workflow orchestration

Core Features:
{features}

Data Pipeline Components:
- Data ingestion from multiple sources (CSV, JSON, APIs, databases)
- Data cleaning and preprocessing
- Feature engineering and transformation
- Statistical analysis and modeling
- Automated report generation
- Data quality checks and validation

Visualization:
- Interactive dashboards with Plotly/Streamlit
- Statistical plots and charts
- Custom visualization functions
- Export capabilities (PNG, PDF, HTML)

Infrastructure:
- Docker containers for reproducible environments
- Environment configuration management
- Data versioning with DVC
- Automated testing for data pipelines
- Logging and monitoring

Documentation:
- Jupyter notebooks with analysis examples
- API documentation for custom functions
- Data dictionary and schema documentation
- Setup and usage instructions

Make it a professional data science project with best practices for reproducibility and scalability.""",
        variables={
            "project_name": "data-analysis-pipeline",
            "features": "- Customer behavior analysis\n- Sales forecasting models\n- A/B testing framework\n- Real-time dashboard\n- Automated reporting system"
        }
    ),
    
    "microservice": ProjectTemplate(
        name="Microservice Architecture",
        description="Scalable microservice with Docker and Kubernetes",
        category="Microservices",
        tags=["microservice", "docker", "kubernetes", "api"],
        prompt_template="""Create a production-ready microservice called '{project_name}' with these specifications:

Technology Stack:
- {language} with appropriate web framework
- PostgreSQL/MongoDB for data persistence
- Redis for caching and sessions
- RabbitMQ/Apache Kafka for messaging
- Docker for containerization

Microservice Features:
{features}

Architecture Requirements:
- Service discovery and registration
- Health checks and readiness probes
- Circuit breaker pattern for resilience
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus
- Structured logging with correlation IDs

API Design:
- RESTful endpoints with OpenAPI documentation
- GraphQL API (if applicable)
- Event-driven communication
- API versioning strategy
- Rate limiting and throttling

Infrastructure:
- Kubernetes deployment manifests
- Helm charts for package management  
- Ingress configuration with SSL
- ConfigMaps and Secrets management
- Horizontal Pod Autoscaling
- Service mesh integration (Istio)

Observability:
- Application metrics and custom dashboards
- Log aggregation with ELK stack
- Distributed tracing visualization
- Alerting rules and notifications

Security:
- JWT-based authentication
- RBAC authorization
- Security scanning in CI/CD
- Network policies

Make it enterprise-ready with full observability, security, and scalability features.""",
        variables={
            "project_name": "user-service",
            "language": "Node.js",
            "features": "- User management and authentication\n- Profile data management\n- Event publishing for user actions\n- Integration with external identity providers\n- Audit logging for compliance"
        }
    )
}

class InteractiveMode:
    """Interactive mode for project generation."""
    
    def __init__(self):
        self.console = console
    
    def show_welcome(self):
        """Show welcome screen."""
        welcome_panel = Panel.fit(
            "[bold cyan]🎯 Welcome to Interactive Mode![/bold cyan]\n\n"
            "Let's create something amazing together!\n"
            "I'll guide you through the process step by step.",
            title="[bold green]🚀 Codegen Interactive[/bold green]",
            border_style="green"
        )
        self.console.print(welcome_panel)
        self.console.print()
    
    def show_templates(self) -> None:
        """Display available templates."""
        table = Table(title="📋 Available Project Templates", box=box.ROUNDED)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta") 
        table.add_column("Description", style="white")
        table.add_column("Tags", style="yellow")
        
        for template in TEMPLATES.values():
            tags_str = ", ".join(template.tags)
            table.add_row(
                template.name,
                template.category,
                template.description,
                tags_str
            )
        
        self.console.print(table)
        self.console.print()
    
    def select_template(self) -> Optional[ProjectTemplate]:
        """Let user select a template."""
        self.show_templates()
        
        template_names = list(TEMPLATES.keys())
        template_display = {k: TEMPLATES[k].name for k in template_names}
        
        while True:
            choice = Prompt.ask(
                "Which template would you like to use?",
                choices=template_names + ["custom", "help"],
                default="webapp"
            )
            
            if choice == "help":
                self.show_template_details()
                continue
            elif choice == "custom":
                return None  # Custom project
            else:
                return TEMPLATES[choice]
    
    def show_template_details(self):
        """Show detailed information about templates."""
        for key, template in TEMPLATES.items():
            panel = Panel.fit(
                f"[bold]{template.description}[/bold]\n\n"
                f"[cyan]Category:[/cyan] {template.category}\n"
                f"[cyan]Tags:[/cyan] {', '.join(template.tags)}\n\n"
                f"[dim]Use: --template {key}[/dim]",
                title=f"[bold yellow]{template.name}[/bold yellow]",
                border_style="blue"
            )
            self.console.print(panel)
        self.console.print()
    
    def customize_template(self, template: ProjectTemplate) -> str:
        """Let user customize template variables."""
        self.console.print(f"[bold cyan]Customizing: {template.name}[/bold cyan]")
        self.console.print()
        
        variables = {}
        
        # Get project name
        variables['project_name'] = Prompt.ask(
            "Project name",
            default=template.variables.get('project_name', 'my-project')
        )
        
        # Template-specific customizations
        if 'features' in template.variables:
            self.console.print("\n[bold]Current features:[/bold]")
            self.console.print(template.variables['features'])
            
            if Confirm.ask("Would you like to customize the features?"):
                features = []
                self.console.print("\n[dim]Enter features (press Enter twice when done):[/dim]")
                while True:
                    feature = Prompt.ask("Feature", default="")
                    if not feature:
                        break
                    features.append(f"- {feature}")
                
                if features:
                    variables['features'] = "\n".join(features)
        
        # Database selection for applicable templates
        if 'database' in template.variables:
            database = Prompt.ask(
                "Database type",
                choices=["PostgreSQL", "MongoDB", "MySQL", "SQLite"],
                default=template.variables['database']
            )
            variables['database'] = database
        
        # Language selection for microservices
        if 'language' in template.variables:
            language = Prompt.ask(
                "Programming language",
                choices=["Node.js", "Python", "Go", "Java"],
                default=template.variables['language']
            )
            variables['language'] = language
        
        return template.render_prompt(variables)
    
    def get_custom_description(self) -> str:
        """Get custom project description from user."""
        self.console.print("[bold cyan]Custom Project Creation[/bold cyan]")
        self.console.print()
        
        self.console.print("[dim]Describe your project in detail. Be specific about:[/dim]")
        self.console.print("[dim]• Technology stack preferences[/dim]")
        self.console.print("[dim]• Key features and functionality[/dim]")
        self.console.print("[dim]• Architecture requirements[/dim]")
        self.console.print("[dim]• Any specific libraries or frameworks[/dim]")
        self.console.print()
        
        description_lines = []
        self.console.print("[dim]Enter your project description (press Enter twice when done):[/dim]")
        
        while True:
            line = input("")
            if not line and description_lines:
                break
            description_lines.append(line)
        
        return "\n".join(description_lines)
    
    def run(self) -> str:
        """Run interactive mode and return project description."""
        self.show_welcome()
        
        # Check if user wants to use a template
        use_template = Confirm.ask("Would you like to use a project template?", default=True)
        
        if use_template:
            template = self.select_template()
            if template:
                return self.customize_template(template)
        
        # Custom project
        return self.get_custom_description()

def get_template_description(template_name: str, variables: Dict[str, str] = None) -> str:
    """Get a template description with optional variable substitution."""
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    
    template = TEMPLATES[template_name]
    return template.render_prompt(variables or {})

def list_templates() -> List[str]:
    """Get list of available template names."""
    return list(TEMPLATES.keys())
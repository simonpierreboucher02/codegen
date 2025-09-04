#!/usr/bin/env python3
"""
Demo script for Codegen CLI - shows the beautiful interface without API calls
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich import box

console = Console()

def demo_banner():
    """Show the beautiful banner"""
    console.print(Panel.fit(
        "[bold cyan]🤖 CODEGEN - AI Code Generator[/bold cyan]\n[dim]Powered by gpt-5 • Production Ready[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()

def demo_generation_details():
    """Show generation details table"""
    info_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    info_table.add_row("[cyan]Description[/cyan]", "[white]Beautiful React todo app with dark mode[/white]")
    info_table.add_row("[cyan]Model[/cyan]", "[yellow]gpt-5[/yellow]")
    info_table.add_row("[cyan]Max Files[/cyan]", "[magenta]100[/magenta]")
    info_table.add_row("[cyan]Timestamp[/cyan]", "[dim]2025-01-15 14:30:22[/dim]")
    
    console.print(Panel(
        info_table,
        title="[bold blue]📄 Generation Details[/bold blue]",
        border_style="blue"
    ))
    console.print()

def demo_project_tree():
    """Show beautiful project tree"""
    tree = Tree(
        "🌳 [bold cyan]react-todo-app[/bold cyan]",
        guide_style="bright_blue"
    )
    
    # Add src folder
    src = tree.add("📁 [bold blue]src[/bold blue]")
    src.add("📄 [green]App.jsx[/green]")
    src.add("📄 [green]index.js[/green]")
    src.add("📄 [green]App.css[/green]")
    
    components = src.add("📁 [bold blue]components[/bold blue]")
    components.add("📄 [green]TodoList.jsx[/green]")
    components.add("📄 [green]TodoItem.jsx[/green]")
    components.add("📄 [green]AddTodo.jsx[/green]")
    
    # Add public folder
    public = tree.add("📁 [bold blue]public[/bold blue]")
    public.add("📄 [green]index.html[/green]")
    public.add("📄 [green]favicon.ico[/green]")
    
    # Add config files
    tree.add("📄 [green]package.json[/green]")
    tree.add("📄 [green]README.md[/green]")
    tree.add("📄 [green].gitignore[/green]")
    
    console.print(Panel(
        tree,
        title="[bold green]Project Structure[/bold green]",
        border_style="green",
        expand=False
    ))
    console.print()
    
    console.print("[bold green]✓[/bold green] Project structure created with [bold]9[/bold] files to generate")
    console.print()

def demo_code_preview():
    """Show syntax highlighted code preview"""
    react_code = '''import React, { useState, useEffect } from 'react';
import './App.css';
import TodoList from './components/TodoList';
import AddTodo from './components/AddTodo';

function App() {
  const [todos, setTodos] = useState([]);
  const [darkMode, setDarkMode] = useState(false);

  // Load todos from localStorage on component mount
  useEffect(() => {
    const savedTodos = localStorage.getItem('todos');
    if (savedTodos) {
      setTodos(JSON.parse(savedTodos));
    }
    
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  // Save todos to localStorage whenever todos change
  useEffect(() => {
    localStorage.setItem('todos', JSON.stringify(todos));
  }, [todos]);

  const addTodo = (text) => {
    const newTodo = {
      id: Date.now(),
      text,
      completed: false
    };
    setTodos([...todos, newTodo]);
  };

  return (
    <div className={`app ${darkMode ? 'dark-mode' : ''}`}>
      <header className="app-header">
        <h1>Todo App</h1>
        <button 
          className="theme-toggle"
          onClick={() => setDarkMode(!darkMode)}
        >
          {darkMode ? '☀️' : '🌙'}
        </button>
      </header>
      
      <main>
        <AddTodo onAdd={addTodo} />
        <TodoList 
          todos={todos} 
          onToggle={toggleTodo}
          onDelete={deleteTodo}
        />
      </main>
    </div>
  );
}

export default App;'''

    syntax = Syntax(
        react_code,
        "jsx",
        theme="monokai",
        line_numbers=True,
        background_color="default"
    )
    
    console.print("[green]✓[/green] [bold]App.jsx[/bold] [dim](2,847 bytes, 67 lines)[/dim]")
    console.print()
    
    console.print(Panel(
        syntax,
        title="[bold cyan]📄 App.jsx[/bold cyan] [dim](jsx)[/dim]",
        border_style="blue",
        expand=False
    ))
    console.print()

def demo_progress_bar():
    """Show progress bar animation"""
    files = [
        "App.jsx",
        "index.js", 
        "App.css",
        "TodoList.jsx",
        "TodoItem.jsx",
        "AddTodo.jsx",
        "index.html",
        "package.json",
        "README.md"
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        task = progress.add_task(
            "[green]Generating files...", 
            total=len(files)
        )
        
        for i, filename in enumerate(files, 1):
            time.sleep(0.3)  # Simulate generation time
            progress.advance(task)
            progress.update(task, description=f"[green]Generated {i}/{len(files)} files")
            
            # Show file created
            console.print(f"[green]✓[/green] [bold]{filename}[/bold] [dim](generated)[/dim]")
        
        time.sleep(0.5)
        progress.update(task, description="[green]Generation complete! ✨")

def demo_summary():
    """Show final summary"""
    console.print()
    
    summary_panel = Panel.fit(
        "[bold green]Project Generated Successfully! 🎉[/bold green]\n\n"
        "[cyan]Project Name:[/cyan] [bold]react-todo-app[/bold]\n"
        "[cyan]Files Generated:[/cyan] [bold green]9[/bold green] / 9 attempted\n"
        "[cyan]Time Elapsed:[/cyan] [bold]4.2s[/bold]\n"
        "[cyan]Location:[/cyan] [bold blue]/Users/developer/react-todo-app[/bold blue]",
        title="[bold green]✓ Generation Complete[/bold green]",
        border_style="green"
    )
    
    console.print(summary_panel)

def main():
    """Run the demo"""
    console.clear()
    
    demo_banner()
    demo_generation_details()
    
    console.print("[bold cyan]🌳 Phase 1: Generating Project Structure[/bold cyan]")
    console.print()
    
    # Simulate API call
    with console.status("[bold green]Calling gpt-5...", spinner="dots"):
        time.sleep(1.5)
    
    demo_project_tree()
    
    console.print("[bold cyan]📝 Phase 2: Generating Files[/bold cyan]")
    console.print()
    
    demo_progress_bar()
    demo_code_preview()
    demo_summary()
    
    console.print("\n[dim]This is a demo - no actual files were generated or API calls made.[/dim]")

if __name__ == "__main__":
    main()
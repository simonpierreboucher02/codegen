# 🖥️ Codegen - AI Code Generator (Production Ready)

A beautiful, hands-free CLI tool that generates complete project structures from simple descriptions using OpenAI's GPT models. Features syntax highlighting, progress bars, and production-ready error handling.

## ✨ Features

- 🎨 **Beautiful Terminal UI** - Rich colors, progress bars, and syntax highlighting
- 📦 **Syntax Highlighting** - Code preview with 30+ supported languages
- ⚡ **Production Ready** - Comprehensive error handling, logging, and configuration
- 🔄 **Real-time Progress** - Live progress tracking with file generation status
- 🎛️ **Configurable** - YAML config file and CLI options for customization
- 🌳 **Visual Project Tree** - Beautiful tree view of generated project structure
- 📊 **Detailed Statistics** - File counts, generation time, and success rates
- 🛡️ **Robust Error Handling** - Graceful failures with detailed error reporting

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd codegen-tool
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## 📖 Usage

### Basic Usage
```bash
python codegen.py \"your project description\"
```

### Advanced Usage with Options
```bash
# Use different model
python codegen.py \"React todo app\" --model gpt-3.5-turbo

# Limit file generation
python codegen.py \"Python CLI tool\" --max-files 20

# Disable code preview for faster generation
python codegen.py \"FastAPI backend\" --no-preview

# Verbose logging for debugging
python codegen.py \"Vue.js website\" --verbose

# Custom preview settings
python codegen.py \"Django blog\" --max-preview-lines 30
```

### CLI Options
```
Options:
  --model TEXT              OpenAI model to use [default: gpt-4]
  --max-files INTEGER       Maximum files to generate [default: 100]
  --no-preview              Disable code preview in terminal
  --max-preview-lines INTEGER  Max lines in code preview [default: 20]
  --verbose / -v            Enable verbose logging
  --help                    Show help message
```

## 🎯 Examples

### Generate an Influencer Website
```bash
python codegen.py \"site web simple pour influenceur\"
```

**Output Preview:**
```
🤖 CODEGEN - AI Code Generator
     Powered by gpt-4 • Production Ready

┌─ 📄 Generation Details ─────────────────┐
│ Description  │ site web simple pour... │
│ Model        │ gpt-4                   │  
│ Max Files    │ 100                     │
│ Timestamp    │ 2024-01-15 14:30:22     │
└─────────────────────────────────────────┘

🌳 Phase 1: Generating Project Structure

┌─ Project Structure ──────────────────────┐
│ 🌳 influencer-site                       │
│ ├── 📁 public                           │
│ │   ├── 📁 assets                       │
│ │   └── 📄 favicon.ico                   │
│ ├── 📁 src                              │
│ │   ├── 📄 index.html                    │
│ │   ├── 📄 about.html                    │
│ │   └── 📁 css                          │
│ └── 📄 README.md                         │
└─────────────────────────────────────────┘

✓ Project structure created with 12 files to generate

📝 Phase 2: Generating Files

⠋ Generated 8/12 files • 0:00:23

✓ styles.css (2,345 bytes, 89 lines)

┌─ 📄 styles.css (css) ───────────────────┐
│  1  /* Base styles */                    │
│  2  * {                                  │
│  3    box-sizing: border-box;            │
│  4    margin: 0;                         │
│  5    padding: 0;                        │
│  6  }                                    │
│  7                                       │
│  8  body {                               │
│  9    font-family: -apple-system...     │
│ 10    background: #0a0a0a;               │
│ ... (79 more lines)                      │
└─────────────────────────────────────────┘
```

### Generate a React Todo App
```bash
python codegen.py \"React todo app with localStorage persistence\"
```

### Generate a Python CLI Tool
```bash
python codegen.py \"Python CLI tool for file organization with click\"
```

## ⚙️ Configuration

### YAML Configuration File
Create a `config.yaml` file to set default values:

```yaml
# Codegen Configuration
model: \"gpt-4\"
max_files: 100
show_code_preview: true
max_preview_lines: 20
log_level: \"INFO\"
theme: \"monokai\"
timeout: 120
```

### Environment Variables
Set configuration via environment variables:

```bash
export CODEGEN_MODEL=\"gpt-3.5-turbo\"
export CODEGEN_MAX_FILES=50
export CODEGEN_SHOW_CODE_PREVIEW=false
```

## 🎨 Supported Languages for Syntax Highlighting

- **Web**: HTML, CSS, SCSS, JavaScript, TypeScript, JSX, TSX, Vue
- **Backend**: Python, Java, C#, PHP, Ruby, Go, Rust
- **Data**: JSON, YAML, XML, SQL, TOML
- **Config**: INI, CONF, ENV
- **Mobile**: Swift, Kotlin, Dart
- **Systems**: C, C++, Bash, PowerShell
- **Markup**: Markdown, reStructuredText

## 📊 What You Get

### Beautiful Terminal Interface
- 🎨 Colorful progress indicators with Rich library
- 📈 Real-time progress bars with time elapsed
- 🌳 Visual project tree display
- 📄 Syntax-highlighted code previews
- 📊 Detailed generation statistics

### Production-Ready Features
- 🛡️ Comprehensive error handling and recovery
- 📝 Structured logging with file and console output
- ⚙️ Flexible configuration system (CLI, env vars, YAML)
- 🔄 Automatic retries for API failures
- 📊 Performance metrics and timing
- 🧪 Input validation and sanitization

### Project Generation
- 📁 Complete project structures with proper organization
- 📄 Full file content with proper formatting
- 🔧 Language-specific configurations and dependencies
- 📚 README files with usage instructions
- ⚙️ Build configurations and environment files

## 🛠️ Development

### Project Structure
```
codegen-tool/
├── codegen.py          # Main CLI application
├── system_prompt.txt   # AI agent system prompt
├── config.yaml        # Default configuration
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── venv/              # Virtual environment
```

### Adding New Features
1. **Language Support**: Add extensions to `get_language_from_extension()`
2. **Themes**: Modify syntax highlighting themes in `show_code_preview()`
3. **Configuration**: Add new options to `CodegenConfig` model
4. **Commands**: Extend the Click CLI with new commands

## 🔧 Troubleshooting

### Common Issues

**API Key Issues**
```
❌ OpenAI API Key Required
Please set your OpenAI API key:
export OPENAI_API_KEY='your-api-key-here'
```

**Generation Fails**
- Check your internet connection
- Verify OpenAI API credits
- Try with `--verbose` flag for detailed logs
- Reduce `--max-files` if hitting rate limits

**Code Preview Issues**
- Use `--no-preview` flag to disable
- Install additional syntax highlighters if needed
- Check terminal color support

### Debug Mode
Run with verbose logging to see detailed information:
```bash
python codegen.py \"your description\" --verbose
```

### Performance Optimization
- Use `gpt-3.5-turbo` for faster (cheaper) generation
- Disable preview with `--no-preview` for large projects
- Adjust `--max-preview-lines` to reduce terminal output

## 📄 License

Open source - free for personal and commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

- 🐛 **Bug Reports**: Create an issue with detailed reproduction steps
- 💡 **Feature Requests**: Describe your use case and requirements  
- 📚 **Documentation**: Help improve examples and documentation
- 🔧 **Code**: Submit PRs for fixes and enhancements

---

**Made with ❤️ and AI** - Transform your ideas into complete projects in seconds!
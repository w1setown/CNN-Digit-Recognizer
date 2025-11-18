# Wiki Index

Complete reference guide to all wiki documentation for the CNN Digit Recognizer project.

## 📚 Documentation Map

### Getting Started & Usage
| Document | Purpose | Audience |
|----------|---------|----------|
| **[Home](Home.md)** | Wiki overview and navigation | Everyone |
| **[Getting Started](Getting-Started.md)** | Installation and setup guide | New users |
| **[User Guide](User-Guide.md)** | How to use the application | End users |
| **[Keyboard Shortcuts](User-Guide.md#keyboard-shortcuts)** | Quick reference | End users |

### Understanding the System
| Document | Purpose | Audience |
|----------|---------|----------|
| **[Architecture Overview](Architecture.md)** | System design and data flow | Developers |
| **[File Structure](File-Structure.md)** | Project organization | Developers |
| **[Core Modules](Core-Modules.md)** | Detailed module documentation | Developers |

### Deep Dives
| Document | Purpose | Audience |
|----------|---------|----------|
| **[GUI Components](GUI-Components.md)** | UI widgets and layout | GUI developers |
| **[Model System](Model-System.md)** | CNN architecture and ensemble | ML developers |
| **[Data Processing](Data-Processing.md)** | Image preprocessing pipeline | Data engineers |
| **[API Documentation](API-Documentation.md)** | Function and class reference | Developers |

### Configuration & Advanced
| Document | Purpose | Audience |
|----------|---------|----------|
| **[Configuration](Configuration.md)** | Settings and customization | Advanced users |
| **[Development Setup](Development-Setup.md)** | Development environment | Contributors |
| **[Troubleshooting](Troubleshooting.md)** | Problem solving | Everyone |

---

## 🎯 Find What You Need

### I want to...

**Install and run the application**
→ Start with [Getting Started](Getting-Started.md)

**Understand how to use the GUI**
→ Read [User Guide](User-Guide.md)

**Understand how the system works**
→ Start with [Architecture Overview](Architecture.md)

**Modify the GUI**
→ Read [GUI Components](GUI-Components.md)

**Modify the model**
→ Read [Model System](Model-System.md)

**Modify preprocessing**
→ Read [Data Processing](Data-Processing.md)

**Set up development environment**
→ Read [Development Setup](Development-Setup.md)

**Fix a problem**
→ Check [Troubleshooting](Troubleshooting.md)

**Learn about a specific module**
→ Check [Core Modules](Core-Modules.md)

**Find a file or directory**
→ Check [File Structure](File-Structure.md)

**Look up an API function**
→ Check [API Documentation](API-Documentation.md)

**Configure settings**
→ Check [Configuration](Configuration.md)

---

## 📖 Reading Paths

### Path 1: Quick Start (30 minutes)
1. [Getting Started](Getting-Started.md) - Installation
2. [User Guide](User-Guide.md) - Basic usage
3. Start drawing digits!

### Path 2: Understanding (2-3 hours)
1. [Architecture Overview](Architecture.md) - System design
2. [Core Modules](Core-Modules.md) - Detailed overview
3. [File Structure](File-Structure.md) - Project organization
4. [Model System](Model-System.md) - AI understanding

### Path 3: Development (5-8 hours)
1. [Development Setup](Development-Setup.md) - Environment setup
2. [Architecture Overview](Architecture.md) - System design
3. [Core Modules](Core-Modules.md) - Code understanding
4. [GUI Components](GUI-Components.md) - UI system
5. [Model System](Model-System.md) - AI system
6. [Data Processing](Data-Processing.md) - Preprocessing
7. [API Documentation](API-Documentation.md) - Function reference

### Path 4: Troubleshooting (variable)
1. [Troubleshooting](Troubleshooting.md) - Common issues
2. Relevant deep-dive documents based on issue

---

## 📋 Document Summaries

### Home.md
- Project overview
- Quick navigation
- Tech stack summary
- Architecture at a glance

### Getting-Started.md
- Prerequisites
- Step-by-step installation
- First run checklist
- Troubleshooting basic issues
- System requirements

### User-Guide.md
- Drawing digits
- Making predictions
- Contributing training data
- Training new models
- Language switching
- Keyboard shortcuts
- Workflow examples
- Result interpretation
- FAQs

### Architecture.md
- System architecture diagram
- Component interaction flow
- Prediction pipeline
- Training pipeline
- Design patterns
- Module dependencies
- Error handling
- Performance considerations
- Extensibility points

### Core-Modules.md
- gui.py - Main application
- model_ensemble.py - Model management
- widgets.py - UI components
- model.py - CNN architecture
- data_utils.py - Data handling
- digit_preprocessing.py - Image preprocessing
- Utility scripts overview
- Import dependencies

### GUI-Components.md
- GUI overview and layout
- DrawingCanvas class
- PredictionDisplay class
- TrainingPanel class
- Main application layout
- Styling and theme
- Responsive design
- Threading patterns
- Accessibility features
- Internationalization

### Model-System.md
- CNN architecture details
- Compilation configuration
- Model ensemble system
- Training pipeline
- Training callbacks
- Data augmentation
- Model storage and files
- Performance metrics
- Adding custom models
- Troubleshooting
- Future improvements

### Data-Processing.md
- Data processing pipeline
- preprocess_digit_image() function
- preprocess_image() function
- load_and_prepare_mnist() function
- MNIST dataset details
- EMNIST dataset details
- Data format specifications
- Image quality considerations
- Data augmentation
- Performance metrics
- Debugging guide

### File-Structure.md
- Complete directory structure
- Directory descriptions
- Utility scripts reference
- Configuration files
- Documentation files
- Key paths and locations
- File size summary
- Files for modifications

### API-Documentation.md
- Complete API reference
- All function signatures
- Parameter documentation
- Return value documentation
- Usage examples
- Error handling

### Configuration.md
- Application settings
- Model configuration
- Training parameters
- UI customization
- Advanced options
- Environment variables

### Development-Setup.md
- Development environment setup
- Git workflow
- Code style guidelines
- Running tests
- Building documentation
- Contributing process
- Debugging tips

### Troubleshooting.md
- Installation issues
- Application launch issues
- Model issues
- Prediction issues
- Training issues
- GUI issues
- Data issues
- Performance issues
- Debugging commands
- Getting more help

---

## 🔍 Quick Reference

### Key Directories
- `src/` - Application code
- `models/` - Trained models
- `assets/` - Images and resources
- `tests/` - Unit tests
- `wiki/` - Documentation

### Main Files
- `run_gui.py` - Start application
- `create_models.py` - Create models
- `requirements.txt` - Dependencies
- `README.md` - Project readme

### Important Classes
- `DigitRecognitionApp` - Main GUI window
- `ModelEnsemble` - Model management
- `DrawingCanvas` - Drawing interface
- `PredictionDisplay` - Results display
- `TrainingPanel` - Training interface

### Key Functions
- `preprocess_digit_image()` - Image preprocessing
- `load_and_prepare_mnist()` - Dataset loading
- `build_cnn_model()` - Model creation
- `ModelEnsemble.predict()` - Predictions

---

## 📊 Documentation Statistics

| Category | Count | Pages |
|----------|-------|-------|
| Getting Started | 2 | Getting-Started, User-Guide |
| Architecture | 2 | Architecture, File-Structure |
| Code Reference | 3 | Core-Modules, API-Documentation, GUI-Components |
| Technical | 3 | Model-System, Data-Processing, Configuration |
| Advanced | 2 | Development-Setup, Troubleshooting |
| **Total** | **12** | **Wiki pages** |

---

## 🔗 Cross-References

### Related to GUI
- [GUI Components](GUI-Components.md) - Widget details
- [User Guide](User-Guide.md) - Usage instructions
- [Architecture Overview](Architecture.md) - System flow
- [Core Modules](Core-Modules.md) - Code structure

### Related to Models
- [Model System](Model-System.md) - Architecture details
- [Core Modules](Core-Modules.md) - Implementation
- [Data Processing](Data-Processing.md) - Input format
- [API Documentation](API-Documentation.md) - Function reference

### Related to Data
- [Data Processing](Data-Processing.md) - Pipeline details
- [Model System](Model-System.md) - Training info
- [Core Modules](Core-Modules.md) - Implementation
- [Troubleshooting](Troubleshooting.md) - Data issues

### Related to Development
- [Development Setup](Development-Setup.md) - Environment setup
- [Architecture Overview](Architecture.md) - System design
- [Core Modules](Core-Modules.md) - Code reference
- [Troubleshooting](Troubleshooting.md) - Problem solving

---

## 📝 Contribution Guide

Want to improve the documentation?

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b docs/my-improvement`
3. **Edit wiki files** in the `wiki/` directory
4. **Follow Markdown formatting:**
   - Use proper heading hierarchy
   - Include code examples
   - Use tables for organization
   - Add cross-references
5. **Test links** before submitting
6. **Submit a pull request**

---

## 🎓 Learning Resources

### External Resources
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Tkinter Tutorial](https://docs.python.org/3/library/tkinter.html)
- [OpenCV Docs](https://docs.opencv.org/)
- [CNN Basics](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

### Project Resources
- [GitHub Repository](https://github.com/w1setown/CNN-Digit-Recognizer)
- [Issues & Discussions](https://github.com/w1setown/CNN-Digit-Recognizer/issues)
- [Project README](../README.md)
- [Model Loading Fixes](../MODEL_LOADING_FIX.md)
- [Reorganization Notes](../REORGANIZATION.md)

---

## 📞 Support

### Getting Help

1. **Check documentation** - Most answers are in the wiki
2. **Search issues** - Others may have faced similar problems
3. **Review troubleshooting** - Common issues and solutions
4. **Run diagnostics** - `python diagnose.py`
5. **Create issue** - With full error details and logs

### Contact

- **Issues:** [GitHub Issues](https://github.com/w1setown/CNN-Digit-Recognizer/issues)
- **Discussions:** [GitHub Discussions](https://github.com/w1setown/CNN-Digit-Recognizer/discussions)

---

## 📄 License & Attribution

This documentation covers the **CNN Digit Recognizer** project.

**Project Repository:** [github.com/w1setown/CNN-Digit-Recognizer](https://github.com/w1setown/CNN-Digit-Recognizer)

---

**Last Updated:** November 2025  
**Wiki Version:** 1.0  
**Documentation Pages:** 12  
**Total Content:** ~25,000 words

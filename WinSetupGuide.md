# Windows Build Guide for HyperTune

This guide will help Windows users set up and build the HyperTune project using the provided Makefile.

## Setup Instructions

### 1. Install MinGW-w64

MinGW-w64 provides the GCC compiler and make utility for Windows.

1. Download the MinGW-w64 installer:
   - Visit [MinGW-w64 Downloads](https://www.mingw-w64.org/downloads/)
   - Recommended: Use the "MinGW-W64-builds" installer

2. Run the installer with these settings:
   - Version: Latest (e.g., 8.1.0)
   - Architecture: x86_64
   - Threads: posix
   - Exception: seh
   - Build revision: Latest
   - Installation location: `C:\mingw-w64` (or your preferred location)

3. Add MinGW to your PATH:
   - Right-click on "This PC" or "My Computer" → Properties
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path", then click "Edit"
   - Add `C:\mingw-w64\mingw64\bin` (or your MinGW installation path + \mingw64\bin)
   - Click "OK" on all dialogs to save changes

4. Verify installation:
   - Open a new Command Prompt
   - Run `g++ --version` and `mingw32-make --version`
   - If both commands show version information, installation was successful

### 2. Project Setup

1. Clone or download the HyperTune project
2. Make sure the Makefile is in the root directory of the project
3. Ensure the following directory structure:
   ```
   HyperTune/
   ├── HyperTune/
   │   └── main.cpp
   ├── openmp/
   │   ├── libgomp-1.dll      (Required for Windows)
   │   ├── libgomp.a          (Required for Windows)
   │   ├── omp.h
   │   └── ... (other OpenMP files)
   └── Makefile
   ```

### 3. Building the Project

1. Open Command Prompt
2. Navigate to the HyperTune project root directory:
   ```
   cd path\to\HyperTune
   ```
3. Run the build command:
   ```
   mingw32-make
   ```
4. If successful, you should see `HyperTune.exe` in the project directory

### 4. Running the Application

Run the application from Command Prompt:
```
HyperTune.exe
```

## Troubleshooting

### "make: command not found"
- Use `mingw32-make` instead of `make`
- Check that MinGW's bin directory is correctly added to your PATH

### "g++: command not found"
- Verify MinGW installation and PATH settings
- Try reinstalling MinGW

### "fatal error: omp.h: No such file or directory"
- Make sure the `openmp` directory exists in the project root
- Check that it contains the required OpenMP header files

### Linker errors about OpenMP
- Ensure `libomp.dll` and `libomp.lib` are in the `openmp` directory
- These files should be compatible with MinGW-w64

### Program runs but crashes when using OpenMP
- Make sure `libomp.dll` is copied to the same directory as your executable
- The Makefile should handle this automatically, but you can also copy it manually

## Alternative: Using Visual Studio

If you prefer using Visual Studio instead of MinGW:

1. Install Visual Studio with C++ development tools
2. Open the project folder in Visual Studio
3. Create a new C++ project
4. Add the existing source files
5. Configure OpenMP support:
   - Project Properties → C/C++ → Language → OpenMP Support: Set to "Yes"
   - Add the `openmp` directory to include paths
   - Add the `openmp` directory to library paths
   - Add `libomp.lib` to additional dependencies


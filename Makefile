# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_60 # Adjust sm_60 if your GPU is newer/older

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = build

# Files
SOURCES = $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SOURCES)))
EXECUTABLE = $(BINDIR)/image_proc

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build successful! Executable is in $(BINDIR)/image_proc"

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

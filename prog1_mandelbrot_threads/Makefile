
CXX=g++ -m64
CXXFLAGS=-I../common -Iobjs/ -O3 -std=c++11 -Wall -fPIC

APP_NAME=mandelbrot
OBJDIR=objs
COMMONDIR=../common

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))

THREADS=2 3 4 5 6 7 8
VIEW=2
OUTPUT_FILE=problem1_view_$(VIEW).txt

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/mandelbrotSerial.o $(OBJDIR)/mandelbrotThread.o $(PPM_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm -lpthread

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/CycleTimer.h

run_all_threads: $(APP_NAME)
	@echo "Running mandelbrot with various thread counts..." > $(OUTPUT_FILE)
	@for t in $(THREADS); do \
		echo "Threads: $$t" >> $(OUTPUT_FILE); \
		./$(APP_NAME) --threads $$t --view $(VIEW) >> $(OUTPUT_FILE); \
	done
	@echo "All results saved to $(OUTPUT_FILE)"

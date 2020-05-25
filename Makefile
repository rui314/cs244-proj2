CXXFLAGS = -std=c++11

poptrie: poptrie.cc p46.o p52.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -Wall -std=c++14 -g -o poptrie -march=native -O3 poptrie.cc p46.o p52.o

clean:
	rm -f *~ *.o poptrie

.PHONY: clean

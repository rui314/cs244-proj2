poptrie: poptrie.cc p52.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -Wall -std=c++14 -g -o poptrie -march=native -O2 poptrie.cc p52.o

clean:
	rm -f *~ *.o poptrie

.PHONY: clean

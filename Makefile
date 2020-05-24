poptrie: poptrie.cc
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -Wall -std=c++14 -g -o poptrie -O2 poptrie.cc

clean:
	rm -f *~ *.o poptrie

.PHONY: clean

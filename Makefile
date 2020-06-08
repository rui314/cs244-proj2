CXXFLAGS = -std=c++11

all: poptrie poptrie-ipv6

poptrie: poptrie.cc p46.o p52.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -Wall -std=c++14 -g -o poptrie -march=native -O3 poptrie.cc p46.o p52.o

poptrie-ipv6: poptrie-ipv6.cc p69-ipv6.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -Wall -std=c++14 -g -o poptrie-ipv6 -march=native poptrie-ipv6.cc p69-ipv6.o

clean:
	rm -f *~ *.o poptrie

.PHONY: all clean

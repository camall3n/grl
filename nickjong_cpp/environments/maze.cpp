#include "maze.h"

#include <cstdio>
#include <cstdlib>

namespace {
    inline unsigned int RandomInteger(unsigned int max) {
	return (unsigned int)(double(rand()) / RAND_MAX * max);
    }
}

Maze::Maze(unsigned int size): junctions(size) {
    std::vector<Coordinate> fringe;
//      junctions.resize(size);
    index.insert(make_pair(Coordinate(0, 0), 0));
    fringe.push_back(Coordinate(-1, 0));
    fringe.push_back(Coordinate(1, 0));
    fringe.push_back(Coordinate(0, -1));
    fringe.push_back(Coordinate(0, 1));

    for (unsigned int newCellIndex = 1; newCellIndex < size; ++newCellIndex) {
	const unsigned int fringeIndex = RandomInteger(fringe.size());
	const Coordinate newCell = fringe[fringeIndex];

	// Debugging code.
//  	for (size_t i = 0; i < fringe.size(); ++i)
//  	    fprintf(stderr, "(%d,%d) ", fringe[i].first, fringe[i].second);
//  	fprintf(stderr, "\n");
//  	for (index_t::const_iterator it = index.begin();
//  	     it != index.end();
//  	     ++it)
//  	{
//  	    const Coordinate &c = it->first;
//  	    fprintf(stderr, "(%d,%d) ", c.first, c.second);
//  	}
//  	fprintf(stderr, "\n");

	fringe[fringeIndex] = fringe.back();
	fringe.pop_back();

	// Debugging code.
//  	fprintf(stderr, "New cell: %d,%d\n", newCell.first, newCell.second);

	for (unsigned i = 0; i < 4; ++i) {
//  	    fprintf(stderr, " dir: %d ", i);

	    Coordinate neighbor = newCell;
	    MoveCoordinate(neighbor, i);
	    index_t::const_iterator it = index.find(neighbor);
	    if (it == index.end()) {
		if (!InFringe(neighbor))
		    fringe.push_back(neighbor);
//  		fprintf(stderr, "new\n");
	    } else {
		junctions[it->second].connect[(i + 2) % 4] = newCellIndex;
		junctions[newCellIndex].connect[i] = it->second;
//  		fprintf(stderr, "old\n");
	    }
	}

//  	std::pair<index_t::const_iterator, bool> retVal =
  	index.insert(make_pair(newCell, newCellIndex));

//  	fprintf(stderr,
//  		retVal.second ? "T (%d,%d)\n" : "F (%d,%d)\n",
//  		retVal.first->first.first, retVal.first->first.second);
    }

//      fprintf(stderr, "Num cells: %d (%d)\n", index.size(), junctions.size());

}

Maze::Maze(FILE *stream) {
    unsigned size;
    if (1 != fscanf(stream, "%d\n", &size))
	fprintf(stderr, "Couldn't read maze size.\n");
    junctions.resize(size);
    for (unsigned i = 0; i < size; ++i) {
	unsigned l[4];
	if (4 != fscanf(stream, "%d %d %d %d \n", &l[0], &l[1], &l[2], &l[3]))
	    fprintf(stderr, "Couldn't read junction %d.\n", i);
	Junction &junction = junctions[i];
	for (unsigned j = 0; j < 4; ++j)
	    junction.connect[j] = l[j];
    }

    std::map<unsigned, Coordinate> reverseIndex;
    reverseIndex[0] = std::make_pair(0,0);
    index[reverseIndex[0]] = 0;
    for (unsigned i = 1; i < size; ++i) {
	const Junction &junction = junctions[i];
	unsigned dir = 0;
	int neighbor;
	do {
	    neighbor = junction.connect[dir++];
	} while (neighbor < 0 || neighbor >= int(i) and dir < 4);
	Coordinate c = reverseIndex[neighbor];
//  	fprintf(stderr, "neighbor %d: (%d, %d)\n",
//  		neighbor, c.first, c.second);
	unsigned reverseDir = (dir + 2) % 4;
	MoveCoordinate(c, reverseDir);
	reverseIndex[i] = c;
	index[c] = i;

//  	fprintf(stderr, "%d: (%d, %d)\n", i, c.first, c.second);
    }
}

Maze::Maze(unsigned height, unsigned width) {
    unsigned size = height * width;
    junctions.resize(size);
    unsigned counter = 0;
    for (unsigned y = 0; y < height; ++y)
	for (unsigned x = 0; x < width; ++x) {
	    Junction &j = junctions[counter];
	    for (unsigned d = 0; d < 4; ++d) {
		bool vertical = d % 2 == 0;
		bool positive = d / 2 == 0;
		int dx = vertical ? 0 : (positive ? 1 : -1);
		int dy = vertical ? (positive ? 1 : -1) : 0;
		int newX = x + dx;
		int newY = y + dy;
		if (newX < 0 || newX >= int(width))
		    continue;
		if (newY < 0 || newY >= int(height))
		    continue;
		unsigned newCounter = newY * width + newX;
		j.connect[d] = newCounter;
	    }

	    index[std::make_pair(x, y)] = counter++;
	}
}

bool Maze::GetView(unsigned int cellIndex, View &out) const {
    if (cellIndex >= junctions.size())
	return false;
    const Junction &j = junctions[cellIndex];
    for (int i = 0; i < 4; ++i)
	out.passable[i] = j.connect[i] >= 0;
    return true;
}

bool Maze::MoveCellIndex(unsigned int &cellIndex, unsigned d) const {
    if (cellIndex >= junctions.size())
	return false;
    int neighbor = junctions[cellIndex].connect[d];
    if (neighbor < 0)
	return false;
    cellIndex = neighbor;
    return true;
}

unsigned int Maze::Size() const {
    return junctions.size();
}

void Maze::CreateGridMap(std::vector<std::vector<bool> > &map) const {
    // Find dimensions of maze.
    int xMin = 0, xMax = 0, yMin = 0, yMax = 0;
    index_t::const_iterator it;
    for (it = index.begin();
	 it != index.end();
	 ++it)
    {
	const Coordinate &pos = it->first;
	if (pos.first < xMin)
	    xMin = pos.first;
	else if (pos.first > xMax)
	    xMax = pos.first;
	if (pos.second < yMin)
	    yMin = pos.second;
	else if (pos.second > yMax)
	    yMax = pos.second;
    }
    int height = yMax - yMin + 1;
    int width = xMax - xMin + 1;
    map.clear();
    map.resize(height, std::vector<bool>(width, false));
    for (it = index.begin();
	 it != index.end();
	 ++it)
    {
	const Coordinate &pos = it->first;
	int x = pos.first - xMin;
	int y = pos.second - yMin;
	map[height - y - 1][x] = true;
    }
}

void Maze::PrintMap(FILE *stream) const {
    std::vector<std::vector<bool> > map;
    CreateGridMap(map);
    for (unsigned i = 0; i < map.size(); ++i) {
	const std::vector<bool> &vec = map[i];
	for (unsigned j = 0; j < vec.size(); ++j)
	    printf(vec[j] ? "x" : " ");
	printf("\n");
    }
}

void Maze::MoveCoordinate(Coordinate &c, unsigned d) {
    switch (d) {
    case 0:
	++c.second; break;
    case 1:
	++c.first; break;
    case 2:
	--c.second; break;
    case 3:
	--c.first; break;
    }
}

void Maze::Serialize(FILE *stream) const {
    unsigned size = Size();
    fprintf(stream, "%d\n", size);
    for (unsigned i = 0; i < size; ++i) {
	const Junction &junction = GetJunction(i);
	for (unsigned j = 0; j < 4; ++j)
	    fprintf(stream, "%d ", junction.connect[j]);
	fprintf(stream, "\n");
    }
}

const Maze::Junction &Maze::GetJunction(unsigned int cellIndex) const {
    return junctions[cellIndex];
}

bool Maze::InFringe(const Coordinate &c) const {
    for (unsigned int d = 0; d < 4; ++d) {
	Coordinate neighbor = c;
	MoveCoordinate(neighbor, d);
	if (index.end() != index.find(neighbor))
	    return true;
    }
    return false;
}

Maze::Junction::Junction() {
    for (int i = 0; i < 4; ++i)
	connect[i] = -1;
}

bool Maze::CompareCoord::operator()(const Coordinate &a,
				    const Coordinate &b) const
{
    const int ax = abs(a.first);
    const int ay = abs(a.second);
    const int bx = abs(b.first);
    const int by = abs(b.second);

    const int manhattan = (ax + ay) - (bx + by);
    if (manhattan < 0)
	return true;
    if (manhattan > 0)
	return false;

    const int xprojection = ax - bx;
    if (xprojection < 0)
	return true;
    if (xprojection > 0)
	return false;

    const bool axpos = a.first == ax;
    const bool bxpos = b.first == bx;
    if (!axpos && bxpos)
	return true;
    if (axpos && !bxpos)
	return false;

    const bool aypos = a.second == ay;
    const bool bypos = b.second == by;
    if (!aypos && bypos)
	return true;

    return false;
}

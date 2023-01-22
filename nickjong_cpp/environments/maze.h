#ifndef _MAZE_H_
#define _MAZE_H_

#include <cstdio>
#include <map>
#include <utility>
#include <vector>

class Maze {
public:
    class Exception {};

    /** Normal coordinate system: first coordinate increases to the
        right, second coodiante increases upwards. */
    typedef std::pair<int,int> Coordinate;

    /** Represents the available moves from a certain cell.  Each bool
        is true iff the corresponding direction is open; the
        directions are in the order NESW. */
    struct View {
	bool passable[4];
	bool operator()(unsigned d) const { return passable[d]; }
    };

    /** Creates a maze with the given number of cells, which must be
        positive. */
    Maze(unsigned int size);

    /** Recreates a maze from one previously serialized. */
    Maze(FILE *stream);

    /** Creates a high entropy (boring) maze consisting of every
        possible cell in a grid with the given dimensions. */
    Maze(unsigned height, unsigned width);

    /** Given a cell index, returns whether or not the position is
        valid and puts available directions to move in 'out'. */
    bool GetView(unsigned int cellIndex, View &out) const;

    /** Change the given cell index by moving in the given direction.
        Returns true iff both the cell index and the direction were
        valid. */
    bool MoveCellIndex(unsigned int &cellIndex, unsigned d) const;

    /** Returns the number of cells in the maze. */
    unsigned int Size() const;

    /** Puts the maze into an occupancy grid like representation */
    void CreateGridMap(std::vector<std::vector<bool> > &map) const;

    /** Prints the maze to the given FILE stream. */
    void PrintMap(FILE *stream) const;

    /** Change the given coordinate by moving in the given direction. */
    static void MoveCoordinate(Coordinate &c, unsigned d);

    /** Serialize the maze to a FILE stream. */
    void Serialize(FILE *stream) const;

protected:
    struct Junction {
	Junction();
	int connect[4];
    };

    const Junction &GetJunction(unsigned int cellIndex) const;
    bool InFringe(const Coordinate &c) const;

private:
    struct CompareCoord {
	bool operator()(const Coordinate &a, const Coordinate &b) const;
    };

    typedef std::map<Coordinate, int, CompareCoord> index_t;

    std::vector<Junction> junctions;
    index_t index;    
};

#endif

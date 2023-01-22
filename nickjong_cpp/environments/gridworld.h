#ifndef _GRIDWORLD_H_
#define _GRIDWORLD_H_

#include "psr.h"
#include "trainer.h"

// This environment models a grid of locations.  The agent can move
// up, down, left, or right.  An observation of 1 indicates that the
// agent attempted to move through a wall; the agent receives an
// observation of 0 otherwise.
class GridWorld: public Environment {
public:
    // Specify the height and width of the grid.  Currently, the agent
    // always begins in the upper left corner.
    GridWorld(int h, int w);

    // Implementations of interface methods.
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

protected:
    // This method determines whether the designated square is a valid
    // square into which to move.
    virtual bool IsValid(int i, int j) const;

    // These methods are intended for subclasses.  Since the relevant
    // data members are primitive types, I don't bother to use these
    // for accessing x and y in GridWorld methods.
    void SetX(int i);
    int GetX() const;
    void SetY(int j);
    int GetY() const;
    int GetHeight() const { return height; }
    int GetWidth() const {return width; }

private:
    const int height, width;
    int x, y;
};

class TestableGridWorld: public TestableEnvironment,
			 public FiniteStateEnvironment,
			 public GridWorld
{
public:
    TestableGridWorld(int h, int w);

    // These three methods merely pass through the still abstract
    // TestableEnvironment interface to the implementations from
    // GridWorld.
    virtual void Reset();
    virtual int NumActions() const;
    virtual int NumObservations() const;

    virtual int ApplyAction(int a);
    virtual const double *GetODist() const;
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    double obsLikelihood[2];
};

class TestableResettableGridWorld: public TestableEnvironment,
				   public GridWorld
{
public:
    TestableResettableGridWorld(int h, int w);

    // These three methods merely pass through the still abstract
    // TestableEnvironment interface to the implementations from
    // GridWorld.
    virtual void Reset();
    virtual int NumActions() const;
    virtual int NumObservations() const;

    virtual int ApplyAction(int a);
    virtual const double *GetODist() const;

private:
    double obsLikelihood[2];
};

#endif

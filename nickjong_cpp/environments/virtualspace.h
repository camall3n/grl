#ifndef _VIRTUALSPACE_H_
#define _VIRTUALSPACE_H_

#include "maze.h"
#include "psr.h"
#include "trainer.h"

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations:
    8 - No visibility (immediate wall)
    otherwise, 3 bit mask of the three (other) possible links in the
     next node
     4 - left
     2 - forward
     1 - right
*/
class VirtualSpace: public TestableEnvironment,
		    public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    VirtualSpace(const Maze *mazeArg);

    virtual ~VirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */

    /** Belief vector over states.  A state index q is formed by 4L +
        O, where L is the grid index and O is the orientation
        (Direction). */
//      double *belief;
    double obsLikelihood[9];
};

/** Actions:
    0 - Move north
    1 - Move east
    2 - Move south
    3 - Move west
    Observations:
    0 - Move succeeded
    1 - Ran into a wall
*/
class OverheadViewBumpSensorVS: public TestableEnvironment,
				public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    OverheadViewBumpSensorVS(const Maze *mazeArg);

    virtual ~OverheadViewBumpSensorVS();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */

    double obsLikelihood[2];
};

/** Actions:
    0 - Move north
    1 - Move east
    2 - Move south
    3 - Move west
    Observations (16):
    one bit for each direction, 1 if wall, 0 if open
    NESW -> Most significant bit to least significant
*/
class OverheadViewOmniSensorVS: public TestableEnvironment,
				public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    OverheadViewOmniSensorVS(const Maze *mazeArg);

    virtual ~OverheadViewOmniSensorVS();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    int lastO; /**< The last observation returned. */

    double obsLikelihood[16];
};

/** Actions:
    0 - Move north
    1 - Move east
    2 - Move south
    3 - Move west
    Observations (16):
    one bit for each direction, 1 if wall, 0 if open
    NESW -> Most significant bit to least significant
    The observations are for the PREVIOUS location.
*/
class OverheadViewDelayedOmniSensorVS: public TestableEnvironment,
				       public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    OverheadViewDelayedOmniSensorVS(const Maze *mazeArg);

    virtual ~OverheadViewDelayedOmniSensorVS();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    int lastO; /**< The last observation returned. */

    double obsLikelihood[16];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations:
    0 - Can move forward
    1 - Can't
*/
class MyopicVirtualSpace: public TestableEnvironment,
			  public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    MyopicVirtualSpace(const Maze *mazeArg);

    virtual ~MyopicVirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */
    double obsLikelihood[2];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations: (16)
    1 bit for each direction, 1 if wall, 0 otherwise
    forward, right, back, left -> MSD to LSD
*/
class RingVirtualSpace: public TestableEnvironment,
			  public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    RingVirtualSpace(const Maze *mazeArg);

    virtual ~RingVirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */
    int lastO;
    double obsLikelihood[16];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations: (16)
    1 bit for each direction, 1 if wall, 0 otherwise
    forward, right, back, left -> MSD to LSD
    Observations delayed one turn.
*/
class DelayedRingVirtualSpace: public TestableEnvironment,
			       public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    DelayedRingVirtualSpace(const Maze *mazeArg);

    virtual ~DelayedRingVirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */
    int lastO;
    double obsLikelihood[16];
};

/** Actions:
    0 - Move north
    1 - Move east
    2 - Move south
    3 - Move west
    Observations (4):
    one bit for each East and West, 1 if wall, 0 if open
    EW -> Most significant bit to least significant
*/
class OverheadViewEWSensorVS: public TestableEnvironment,
			      public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    OverheadViewEWSensorVS(const Maze *mazeArg);

    virtual ~OverheadViewEWSensorVS();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    int lastO; /**< The last observation returned. */

    double obsLikelihood[16];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations:
    8 - No visibility (immediate wall)
    otherwise, 3 bit mask of the three (other) possible links in the
     next node
     4 - left
     2 - forward
     1 - right
    Note: OBSERVATIONS DELAYED ONE ROUND
*/
class DelayedVirtualSpace: public TestableEnvironment,
		    public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    DelayedVirtualSpace(const Maze *mazeArg);

    virtual ~DelayedVirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */
    unsigned lastO;       /**< The last value returned by ApplyAction. */

    /** Belief vector over states.  A state index q is formed by 4L +
        O, where L is the grid index and O is the orientation
        (Direction). */
//      double *belief;
    double obsLikelihood[9];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations:
    0 - Can move forward
    1 - Can't
    Observations delayed one step!
*/
class DelayedMyopicVirtualSpace: public TestableEnvironment,
				 public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    DelayedMyopicVirtualSpace(const Maze *mazeArg);

    virtual ~DelayedMyopicVirtualSpace();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned orientation; /** Actual orientation. */
    double obsLikelihood[2];
};

/** Actions:
    0 - Move forward
    1 - Turn left
    2 - Turn right
    Observations:
    8 - No visibility (immediate wall)
    otherwise, 3 bit mask of the three (other) possible links in the
     next node
     4 - left
     2 - forward
     1 - right
    Note: OBSERVATIONS DELAYED ONE ROUND
*/
class DelayedVisualGridWorldVS: public TestableEnvironment,
				public FiniteStateEnvironment
{
public:
    /** Does not assume ownership of mazeArg. */
    DelayedVisualGridWorldVS(const Maze *mazeArg);

    virtual ~DelayedVisualGridWorldVS();

    // Implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // Implementation of TestableEnvironment methods
    virtual const double *GetODist() const;

    // Implementation of FiniteStateEnvironment methods
    virtual void SetState(unsigned q);
    virtual unsigned NumStates() const;

private:
    const Maze *maze;
    unsigned location;    /** Actual grid index. */
    unsigned lastO;       /**< The last value returned by ApplyAction. */

    /** Belief vector over states.  A state index q is formed by 4L +
        O, where L is the grid index and O is the orientation
        (Direction). */
//      double *belief;
    double obsLikelihood[9];
};

#endif

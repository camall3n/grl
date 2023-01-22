#ifndef _FLOATRESET_H_
#define _FLOATRESET_H_

#include "psr.h"
#include "trainer.h"

class FloatReset: public Environment {
public:
    /* init determines the initial state.  If init == -1, then the
       initial state is random, and each Reset randomly chooses a
       state.  (The states are numbered 0 to 4 in increasing distance
       from the reset state.) */
    FloatReset(int init = -1);

    // implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    int ExamineState() const;

private:
    int initState;
    int state;
};

// Another implementation of the same environment that can also
// generate the true one-step observation probabilities.
class TestableFloatReset: public TestableEnvironment {
public:
    TestableFloatReset(double rNoiseArg, double oNoiseArg, int init = -1);

    // implementations of Environment methods
    virtual void Reset();
    virtual int ApplyAction(int a);
    virtual int NumActions() const;
    virtual int NumObservations() const;

    // implementation of LearnableEnvironment method
    virtual const double *GetODist() const;

private:
    double resetNoise;
    double otherNoise;
    int initState;
    double beliefState[5];
    double obsLikelihood[2];
};

#endif

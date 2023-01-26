#include <cstdlib>
#include "floatreset.h"

FloatReset::FloatReset(int init): initState(init) {
    Reset();
}

void FloatReset::Reset() {
    if (initState >= 0 && initState < 5)
	state = initState;
    else
	state = rand() % 5;
}

int FloatReset::ApplyAction(int a) {
    int delta;
    int retval;
    switch (a) {
    case 0:
	delta = (rand() % 2) * 2 - 1; // 1 or -1
	state += delta;
	if (state < 0)
	    state = 0;
	if (state > 4)
	    state = 4;
	retval = 0;
	break;
    case 1:
	retval = (state == 0) ? 1 : 0;
	state = 0;
	break;
    default:
	throw Exception();
    }
    return retval;
}

int FloatReset::NumActions() const {
    return 2;
}

int FloatReset::NumObservations() const {
    return 2;
}

int FloatReset::ExamineState() const {
    return state;
}

TestableFloatReset::TestableFloatReset(double rNoiseArg,
				       double oNoiseArg,
				       int init)
    : resetNoise(rNoiseArg), otherNoise(oNoiseArg), initState(init)
{
    Reset();
}

void TestableFloatReset::Reset() {
    int i;
    if (initState >= 0 && initState < 5) {
	for (i = 0; i < 5; ++i)
	    beliefState[i] = 0.0;
	beliefState[initState] = 1.0;
    } else 
	for (i = 0; i < 5; ++i)
	    beliefState[i] = 0.2;
}

int TestableFloatReset::ApplyAction(int a) {
    int i;
    double buffer[5];
    double prob = double(rand()) / RAND_MAX;

    switch (a) {
    case 0:
	obsLikelihood[1] = 0.0;
	obsLikelihood[0] = 1.0;
	buffer[0] = 0.5 * (beliefState[0] + beliefState[1]);
	for (i = 1; i < 4; ++i)
	    buffer[i] = 0.5 * (beliefState[i - 1] + beliefState[i + 1]);
	buffer[4] = 0.5 * (beliefState[3] + beliefState[4]);
	for (i = 0; i < 5; ++i)
	    beliefState[i] = buffer[i];
	break;
    case 1:
	obsLikelihood[1] = beliefState[0] * (1.0 - resetNoise)
	    + (1.0 - beliefState[0]) * otherNoise;
	obsLikelihood[0] = 1.0 - obsLikelihood[1];
	beliefState[0] = 1.0;
	for (i = 1; i < 5; ++i)
	    beliefState[i] = 0.0;
	break;
    default:
	throw Exception();
    }

    return (prob <= obsLikelihood[0]) ? 0 : 1;
}

int TestableFloatReset::NumActions() const {
    return 2;
}

int TestableFloatReset::NumObservations() const {
    return 2;
}

const double *TestableFloatReset::GetODist() const {
    return obsLikelihood;
}

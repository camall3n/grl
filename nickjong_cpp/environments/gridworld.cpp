#include "gridworld.h"

GridWorld::GridWorld(int h, int w): height(h), width(w), x(0), y(0) {}

void GridWorld::Reset() {
    x = 0;
    y = 0;
}

int GridWorld::ApplyAction(int a) {
    int dx = 0;
    int dy = 0;

    switch (a) {
    case 0: dy = -1; break;
    case 1: dy =  1; break;
    case 2: dx = -1; break;
    case 3: dx =  1; break;
    default: throw Exception();
    }
 
    int newX = x + dx;
    int newY = y + dy;
    bool okay = IsValid(newX, newY);
    if (okay) {
	x = newX;
	y = newY;
    }

    return okay ? 0 : 1;
}

int GridWorld::NumActions() const {
    return 4;
}

int GridWorld::NumObservations() const {
    return 2;
}

bool GridWorld::IsValid(int i, int j) const {
    return i >= 0 && j >= 0 && i < width && j < height;
}

void GridWorld::SetX(int i) {
    x = i;
}

int GridWorld::GetX() const {
    return x;
}

void GridWorld::SetY(int j) {
    y = j;
}

int GridWorld::GetY() const {
    return y;
}

TestableGridWorld::TestableGridWorld(int h, int w): GridWorld(h, w) {}

int TestableGridWorld::ApplyAction(int a) {
    int o = GridWorld::ApplyAction(a);
    obsLikelihood[o] = 1.0;
    obsLikelihood[1 - o] = 0.0;
    return o;
}

const double *TestableGridWorld::GetODist() const {
    return obsLikelihood;
}

void TestableGridWorld::SetState(unsigned q) {
    SetY(q / GetWidth());
    SetX(q % GetWidth());
}

unsigned TestableGridWorld::NumStates() const {
    return GetHeight() * GetWidth();
}

void TestableGridWorld::Reset() {
    GridWorld::Reset();
}

int TestableGridWorld::NumActions() const {
    return GridWorld::NumActions();
}

int TestableGridWorld::NumObservations() const {
    return GridWorld::NumObservations();
}

TestableResettableGridWorld::TestableResettableGridWorld(int h, int w)
    : GridWorld(h, w)
{}

int TestableResettableGridWorld::ApplyAction(int a) {
    int o;
    if (a < 4)
	o = GridWorld::ApplyAction(a);
    else {
	Reset();
	o = 0;
    }
    obsLikelihood[o] = 1.0;
    obsLikelihood[1 - o] = 0.0;
    return o;
}

const double *TestableResettableGridWorld::GetODist() const {
    return obsLikelihood;
}

void TestableResettableGridWorld::Reset() {
    GridWorld::Reset();
}

int TestableResettableGridWorld::NumActions() const {
    return 5;
}

int TestableResettableGridWorld::NumObservations() const {
    return GridWorld::NumObservations();
}

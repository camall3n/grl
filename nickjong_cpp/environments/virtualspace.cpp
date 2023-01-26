#include <cstdio>

#include "virtualspace.h"

VirtualSpace::VirtualSpace(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

VirtualSpace::~VirtualSpace () {}

void VirtualSpace::Reset() {
    location = orientation = 0;
}

int VirtualSpace::ApplyAction(int a) {
    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    unsigned o = 0;
    Maze::View v;
    maze->GetView(location, v);
    if (!v(orientation))
	o = 8;
    else {
	unsigned forwardLoc = location;
	maze->MoveCellIndex(forwardLoc, orientation);
	maze->GetView(forwardLoc, v);
	for (unsigned i = 0; i < 3; ++i) {
	    unsigned dir = (orientation + 3 + i) % 4;
	    o *= 2;
	    if (v(dir))
		++o;
	}
    }

    for (unsigned i = 0; i < 9; ++i)
	obsLikelihood[i] = 0.0;
    obsLikelihood[o] = 1.0;

    return o;
}

int VirtualSpace::NumActions() const {
    return 3;
}

int VirtualSpace::NumObservations() const {
    return 9;
}

const double *VirtualSpace::GetODist() const {
    return obsLikelihood;
}

void VirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned VirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

OverheadViewBumpSensorVS::
OverheadViewBumpSensorVS(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

OverheadViewBumpSensorVS::~OverheadViewBumpSensorVS () {}

void OverheadViewBumpSensorVS::Reset() {
    location = 0;
}

int OverheadViewBumpSensorVS::ApplyAction(int a) {
    bool okay = maze->MoveCellIndex(location, a);
    int o = okay ? 0 : 1;
    obsLikelihood[o] = 1.0;
    obsLikelihood[1 - o] = 0.0;

    return o;
}

int OverheadViewBumpSensorVS::NumActions() const {
    return 4;
}

int OverheadViewBumpSensorVS::NumObservations() const {
    return 2;
}

const double *OverheadViewBumpSensorVS::GetODist() const {
    return obsLikelihood;
}

void OverheadViewBumpSensorVS::SetState(unsigned q) {
    location = q;
}

unsigned OverheadViewBumpSensorVS::NumStates() const {
    return maze->Size();
}

OverheadViewOmniSensorVS::
OverheadViewOmniSensorVS(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

OverheadViewOmniSensorVS::~OverheadViewOmniSensorVS () {}

void OverheadViewOmniSensorVS::Reset() {
    location = 0;
    lastO = 0;
    for (unsigned i = 0; i < 16; ++i)
	obsLikelihood[i] = 0.0;
}

int OverheadViewOmniSensorVS::ApplyAction(int a) {
    maze->MoveCellIndex(location, a);
    Maze::View v;
    maze->GetView(location, v);

    int o = 0;
    for (unsigned i = 0; i < 4; ++i) {
	o *= 2;
	if (!v(i))
	    ++o;
    }
    
    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    return o;
}

int OverheadViewOmniSensorVS::NumActions() const {
    return 4;
}

int OverheadViewOmniSensorVS::NumObservations() const {
    return 16;
}

const double *OverheadViewOmniSensorVS::GetODist() const {
    return obsLikelihood;
}

void OverheadViewOmniSensorVS::SetState(unsigned q) {
    location = q;
}

unsigned OverheadViewOmniSensorVS::NumStates() const {
    return maze->Size();
}

OverheadViewDelayedOmniSensorVS::
OverheadViewDelayedOmniSensorVS(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

OverheadViewDelayedOmniSensorVS::~OverheadViewDelayedOmniSensorVS () {}

void OverheadViewDelayedOmniSensorVS::Reset() {
    location = 0;
    lastO = 0;
    for (unsigned i = 0; i < 16; ++i)
	obsLikelihood[i] = 0.0;
}

int OverheadViewDelayedOmniSensorVS::ApplyAction(int a) {
    Maze::View v;
    maze->GetView(location, v);

    int o = 0;
    for (unsigned i = 0; i < 4; ++i) {
	o *= 2;
	if (!v(i))
	    ++o;
    }
    
    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    maze->MoveCellIndex(location, a);

    return o;
}

int OverheadViewDelayedOmniSensorVS::NumActions() const {
    return 4;
}

int OverheadViewDelayedOmniSensorVS::NumObservations() const {
    return 16;
}

const double *OverheadViewDelayedOmniSensorVS::GetODist() const {
    return obsLikelihood;
}

void OverheadViewDelayedOmniSensorVS::SetState(unsigned q) {
    location = q;
}

unsigned OverheadViewDelayedOmniSensorVS::NumStates() const {
    return maze->Size();
}

MyopicVirtualSpace::MyopicVirtualSpace(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

MyopicVirtualSpace::~MyopicVirtualSpace () {}

void MyopicVirtualSpace::Reset() {
    location = orientation = 0;
}

int MyopicVirtualSpace::ApplyAction(int a) {
    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    Maze::View v;
    maze->GetView(location, v);
    int o = v(orientation) ? 0 : 1;
    obsLikelihood[1 - o] = 0.0;
    obsLikelihood[o] = 1.0;

    return o;
}

int MyopicVirtualSpace::NumActions() const {
    return 3;
}

int MyopicVirtualSpace::NumObservations() const {
    return 2;
}

const double *MyopicVirtualSpace::GetODist() const {
    return obsLikelihood;
}

void MyopicVirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned MyopicVirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

RingVirtualSpace::RingVirtualSpace(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

RingVirtualSpace::~RingVirtualSpace () {}

void RingVirtualSpace::Reset() {
    location = orientation = 0;
    lastO = 0;
    for (unsigned i = 0; i < 16; ++i)
	obsLikelihood[i] = 0.0;
}

int RingVirtualSpace::ApplyAction(int a) {
    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    Maze::View v;
    maze->GetView(location, v);
    int o = 0;
    for (unsigned i = 0; i < 4; ++i) {
	unsigned dir = (orientation + i) % 4;
	o *= 2;
	if (v(dir))
	    ++o;
    }
    
    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    return o;
}

int RingVirtualSpace::NumActions() const {
    return 3;
}

int RingVirtualSpace::NumObservations() const {
    return 16;
}

const double *RingVirtualSpace::GetODist() const {
    return obsLikelihood;
}

void RingVirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned RingVirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

DelayedRingVirtualSpace::DelayedRingVirtualSpace(const Maze *mazeArg)
    : maze(mazeArg)
{
    Reset();
}

DelayedRingVirtualSpace::~DelayedRingVirtualSpace () {}

void DelayedRingVirtualSpace::Reset() {
    location = orientation = 0;
    lastO = 0;
    for (unsigned i = 0; i < 16; ++i)
	obsLikelihood[i] = 0.0;
}

int DelayedRingVirtualSpace::ApplyAction(int a) {
    Maze::View v;
    maze->GetView(location, v);
    int o = 0;
    for (unsigned i = 0; i < 4; ++i) {
	unsigned dir = (orientation + i) % 4;
	o *= 2;
	if (v(dir))
	    ++o;
    }
    
    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    return o;
}

int DelayedRingVirtualSpace::NumActions() const {
    return 3;
}

int DelayedRingVirtualSpace::NumObservations() const {
    return 16;
}

const double *DelayedRingVirtualSpace::GetODist() const {
    return obsLikelihood;
}

void DelayedRingVirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned DelayedRingVirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

OverheadViewEWSensorVS::
OverheadViewEWSensorVS(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

OverheadViewEWSensorVS::~OverheadViewEWSensorVS () {}

void OverheadViewEWSensorVS::Reset() {
    location = 0;
    lastO = 0;
    for (unsigned i = 0; i < 16; ++i)
	obsLikelihood[i] = 0.0;
}

int OverheadViewEWSensorVS::ApplyAction(int a) {
    maze->MoveCellIndex(location, a);
    Maze::View v;
    maze->GetView(location, v);

    int o = 0;
    for (unsigned i = 1; i < 4; i += 2) {
	o *= 2;
	if (!v(i))
	    ++o;
    }
    
    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    return o;
}

int OverheadViewEWSensorVS::NumActions() const {
    return 4;
}

int OverheadViewEWSensorVS::NumObservations() const {
    return 4;
}

const double *OverheadViewEWSensorVS::GetODist() const {
    return obsLikelihood;
}

void OverheadViewEWSensorVS::SetState(unsigned q) {
    location = q;
}

unsigned OverheadViewEWSensorVS::NumStates() const {
    return maze->Size();
}

DelayedVirtualSpace::DelayedVirtualSpace(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

DelayedVirtualSpace::~DelayedVirtualSpace () {}

void DelayedVirtualSpace::Reset() {
    location = orientation = 0;
    for (unsigned i = 0; i < 9; ++i)
	obsLikelihood[i] = 0.0;
    lastO = 0;
}

int DelayedVirtualSpace::ApplyAction(int a) {
    unsigned o = 0;
    Maze::View v;
    maze->GetView(location, v);
    if (!v(orientation))
	o = 8;
    else {
	unsigned forwardLoc = location;
	maze->MoveCellIndex(forwardLoc, orientation);
	maze->GetView(forwardLoc, v);
	for (unsigned i = 0; i < 3; ++i) {
	    unsigned dir = (orientation + 3 + i) % 4;
	    o *= 2;
	    if (v(dir))
		++o;
	}
    }

    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    return o;
}

int DelayedVirtualSpace::NumActions() const {
    return 3;
}

int DelayedVirtualSpace::NumObservations() const {
    return 9;
}

const double *DelayedVirtualSpace::GetODist() const {
    return obsLikelihood;
}

void DelayedVirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned DelayedVirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

DelayedMyopicVirtualSpace::DelayedMyopicVirtualSpace(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

DelayedMyopicVirtualSpace::~DelayedMyopicVirtualSpace () {}

void DelayedMyopicVirtualSpace::Reset() {
    location = orientation = 0;
}

int DelayedMyopicVirtualSpace::ApplyAction(int a) {
    Maze::View v;
    maze->GetView(location, v);
    int o = v(orientation) ? 0 : 1;
    obsLikelihood[1 - o] = 0.0;
    obsLikelihood[o] = 1.0;

    switch (a) {
    case 0: // move forward
	maze->MoveCellIndex(location, orientation);
	break;
    case 1: // turn left
	orientation += 3;
	orientation %= 4;
	break;
    case 2: // turn right
	orientation += 1;
	orientation %= 4;
	break;
    }

    return o;
}

int DelayedMyopicVirtualSpace::NumActions() const {
    return 3;
}

int DelayedMyopicVirtualSpace::NumObservations() const {
    return 2;
}

const double *DelayedMyopicVirtualSpace::GetODist() const {
    return obsLikelihood;
}

void DelayedMyopicVirtualSpace::SetState(unsigned q) {
    location = q / 4;
    orientation = q % 4;
}

unsigned DelayedMyopicVirtualSpace::NumStates() const {
    return maze->Size() * 4;
}

DelayedVisualGridWorldVS::DelayedVisualGridWorldVS(const Maze *mazeArg): maze(mazeArg) {
    Reset();
}

DelayedVisualGridWorldVS::~DelayedVisualGridWorldVS () {}

void DelayedVisualGridWorldVS::Reset() {
    location = 0;
    for (unsigned i = 0; i < 9; ++i)
	obsLikelihood[i] = 0.0;
    lastO = 0;
}

int DelayedVisualGridWorldVS::ApplyAction(int a) {
    unsigned o = 0;
    unsigned orientation = a;
    Maze::View v;
    maze->GetView(location, v);
    if (!v(orientation))
	o = 8;
    else {
	unsigned forwardLoc = location;
	maze->MoveCellIndex(forwardLoc, orientation);
	maze->GetView(forwardLoc, v);
	for (unsigned i = 0; i < 3; ++i) {
	    unsigned dir = (orientation + 3 + i) % 4;
	    o *= 2;
	    if (v(dir))
		++o;
	}
    }

    obsLikelihood[lastO] = 0.0;
    obsLikelihood[o] = 1.0;
    lastO = o;

    maze->MoveCellIndex(location, orientation);

    return o;
}

int DelayedVisualGridWorldVS::NumActions() const {
    return 4;
}

int DelayedVisualGridWorldVS::NumObservations() const {
    return 9;
}

const double *DelayedVisualGridWorldVS::GetODist() const {
    return obsLikelihood;
}

void DelayedVisualGridWorldVS::SetState(unsigned q) {
    location = q;
}

unsigned DelayedVisualGridWorldVS::NumStates() const {
    return maze->Size();
}

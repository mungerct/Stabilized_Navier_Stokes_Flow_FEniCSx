SetFactory("OpenCASCADE");

// Geometry
r = 0.5*0.1;
cx = 0.2;
cy = 0.2;
L = 2.2;
W = 0.41;
//H = 0.125;

// Element parameters
/*
lc_coarse = 0.1*W;
lc_med = 0.015*W;
lc_fine = 0.05*r;
lc_super_fine = 0.03*r;
lc_ultra_fine = 0.1*lc_super_fine;
*/

lc_coarse = 0.5*W;
lc_med = 0.05*W/12;
lc_fine = 0.1*r;
lc_super_fine = 0.06*r/2;
lc_ultra_fine = 0.5*lc_super_fine/8;

x_min = 0;
x_max = x_min + L;

y_min = 0;
y_max = W;

// Build geometry
Rectangle(1) = {x_min, y_min,  0, L, W, 0};
Circle(5) = {cx, cy, 0, r, 0, 2*Pi};

Curve Loop(2) = {3, 4, 1, 2};
Curve Loop(3) = {5};

// Form plane with hole for cylinder
Plane Surface(2) = {2, 3};

Recursive Delete {
  Curve{3}; 
}
Recursive Delete {
  Surface{1}; 
}

// Set element size over entire domain
Field[1] = Box;
Field[1].VIn = lc_med;
Field[1].VOut = lc_coarse;
Field[1].XMax = x_max*1.1;
Field[1].XMin = x_min*1.1;
Field[1].YMax = y_max*1.1;
Field[1].YMin = y_min*1.1;

// Set refined element size near cylinder
Field[2] = Box;
Field[2].VIn = lc_fine;
Field[2].VOut = lc_coarse;
Field[2].XMax = cx + r*5;
Field[2].XMin = cx - r*1.5;
Field[2].YMax = y_max*1.1;
Field[2].YMin = y_min*1.1;

// Set more refined element size near cylinder
Field[3] = Box;
Field[3].VIn = lc_super_fine;
Field[3].VOut = lc_coarse;
Field[3].XMax = cx + r*2.5;
Field[3].XMin = cx - r*1.5;
Field[3].YMax = cy + r*2.5;
Field[3].YMin = cy - r*2.5;

Field[4] = Cylinder;
Field[4].Radius = r*1.1;
Field[4].VIn = lc_ultra_fine;
Field[4].VOut = lc_coarse;
Field[4].XAxis = 0;
Field[4].XCenter = cx;
Field[4].YAxis = 0;
Field[4].YCenter = cy;
Field[4].ZAxis = 1;


// Set background mesh
Field[5] = Min;
Field[5].FieldsList = {1, 2, 3, 4};
Background Field = 5;

Mesh.Algorithm = 8;
// Recombine Surface{2};

Physical Surface("fluid") = {2};
Physical Curve("inlet") = {4};
Physical Curve("outlet") = {2};
Physical Curve("walls") = {1,3};
Physical Curve("obstacle") = {5};


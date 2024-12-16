# Stabilized Stokes Flow FEniCSx

This is a repo for Stabilized Stokes Flow using FEniCSx by Caleb Munger.

This repo contains code for FEniCSx 0.9.0.0 that takes a black and white .png image as the inlet shape, and than computes the outlet shape (example below)
post prcoessing the image to get the flow profile at the inlet and outlet is done using paraview 12.1. This code can be used to predict the shape of 
of a fluid at an outlet (assuming no mixing) for low Reynolds number exetrusion flow. 

<details>
<summary>Overview</summary>
  The solver uses stabilized stokes flow to solve for the outlet profile. To run the "StokesChannelFlow.py" file, you must also download the the "image2gmsh3D.py" and 
  "image2inlet.py" files. You will also need a black and white .png file of the inlet shape (see "Plus.png" as an example). The second required input is the flowrate ratio 
  between the inner and outer flow profiles. A flowrate ratio of 1 means all of the flow will come from the inner countour, while 0 means all of the flow is in the outer contour.


  There is a third optional parameter which is the length of the mesh elements (the small the number, the more elements). It is recommend to start with a mesh length input of 0.1
  and gradually decrease from there to a suitable resolution. Please know that there is a currently a bug where as the mesh gets smaller, a presssure singularity will form and the
  pressure at the inlet will go towards infinity, this is why starting with a mesh length of 0.1 is recommended.

  
  The examples generated below used a flowrate ratio 0.5
</details>


<details>
<summary>Inlet Example</summary>
<br>
  This is an example of the inlet profile used
  
  ![Plus](Pics/Plus.png)
</details>

<details>
<summary>Inlet Profile Example</summary>
<br>
  This is an example of the inlet profile streamtrace
  
  ![InletShapePlus](Pics/InletShapePlus.png)
</details>

<details>
<summary>Outlet Profile Example</summary>
<br>
  This is an example of the outlet profile streamtrace 
  
  ![InletShapePlus](Pics/OutletShapePlus.png)
</details>

<details>
<summary>Duct Stokes Flow</summary>
<br>
  The "DuctStokesFlow.py" is meant as a test file. Its inputs are where you want to name the gmsh mesh file (Ex: "DuctMesh"), the length of the mesh elements (this is the same as above,
  it is recommened to start with 0.1), and the length of the total domain.


  The duct flow simulations stokes flow through a square cross-section duct, with no obstructions. If the length of the duct is long enough (4 should be enough) the outlet profile will be 
  fully devolped channel flow.


  If you want to use the "StokesChannelFlow.py" file, it is recommended you first run the "DuctStokesFlow.py" file to make sure you have everything set up correctly (note, there are extra packages 
  you will need to install when running the "StokesChannelFlow.py"), but the duct flow is computational easier to run, and has a known output.
</details>

<details>
<summary>Stabilization Methods</summary>
<br>
  The solver can be with 2 differnet element types to produce a stable output. P2-P1 (quadratic velocity and linear pressure) elements, also called "Taylor-Hood" elements are a stable mixed formulation element pair.
  The other set of elements that can be used are stabilized P1-P1 (linear velocity and pressure). Grad-Div stabilization is used to allow the use of P1-P1 elements. Grad-Div stabilization worked by ensuring
  the FE stiffness matrix is positive definite, ensuring a solution exists.
</details>

<details>
<summary>Stream Tracing</summary>
<br>
  To generate a stream-trace of the output, there are 2 files in the Stokes FLow Folder. The first file that must be used is the "reverse_streamtrace_xdmf.py" file. This is not a standard python file, this is a pvpython file.
  pvpython is a Paraview version of python that can be downloaded onlined, if you have installed paraview, most likely pythonpy is also in the paraview bin. To run the "reverse_streamtrace_xdmf.py" the inputs are the .xdmf velocity file made
  from running "StokesChannelFlow.py", when calling the .xxmdf, the absolute location of the file path must be given, even if the "reverse_streamtrace_xdmf.py" and .xdmf file are in the same folder. The next 2 inputs are the number of seeds for 
  the reverese stream trace, it is recommended to start with 200 by 200. The last input is the name of a text file the reverese stream trace seeds will be written to, ex: "reversest.txt"


  To turn the reverse stream trace into .png image of the inlet and outlet as shown above, the "process_streamtrace.py" file is required. This is a regular python file with inputs of the png image of the inlet shape, the reverse stream trace text 
  file ex: "reversest.txt", and the number of seeds. The number of seeds must match the number of seeds used in the "reverse_streamtrace_xdmf.py" file.
</details>

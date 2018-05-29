_tflearn module needs some extra supports, like curses and so on,
 but pip cannot install this immediately,
  but we can do like this:_ 
<br/>
  _I was able to run tflearn with GPU support on windows 10 - 64 bit. Here are the steps to install.
<br/>
Install Python 3.5 (64 bit) from Anaconda. https://repo.continuum.io/archive/Anaconda3-4.2.0-Windows-x86_64.exe (Distribution from python.org does not work due to missing hdf5 module)
<br/>
Install CUDA8 from nvidia site https://developer.nvidia.com/cuda-downloads
<br/>
Install tensorflow with GPU support using command pip install tensorflow-gpu
<br/>
Then go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#curses download curses‑2.2‑cp35‑none‑win_amd64.whl file and run pip install curses‑2.2‑cp35‑none‑win_amd64.whl command._
<br/>
`pip install .\extraLibs\curses-2.2-cp36-cp36m-win_amd64.whl`
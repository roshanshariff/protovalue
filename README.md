The Protovaluefunctionator
==========================

A tool to visualize proto-value functions (see below).  Requires a sufficiently new Python 3 with NumPy and Matplotlib.

The user interface is mostly self-explanatory: use the left mouse button to draw walls in the grid world and the right mouse button to erase them.  The reset button clears all walls.  The sliders select the proto-value function to display; the first by index and the second by eigenvalue.

---

This is the abstract of the talk for which this program was developed:

What Are Proto-Value Functions and Where Do They Come From?
-----------------------------------------------------------

Value functions for MDPs are often "smooth", in that neighbouring states have similar values.  They are therefore well-approximated by linear combinations of smooth "basis" functions.  Tile coding, for example, uses basis functions that generalize amongst states with similar numerical representations.  By ignoring the structure of the MDP, however, a tile coding agent may conflate the values of states that are far apart and yet happen to be represented similarly, and vice versa.

The proto-value functions (PVFs) of Mahadevan and Maggioni (2007) are smooth in a different way: states have similar values if a short random walk can travel between them.  This talk introduces PVFs and how they arise from "diffusion operators" on the state space of a Markov chain. To supplement the paper, I will sketch an intuitive picture of heat diffusing through the state space which motivates two complementary facets of what PVFs are:
 * a notion of smoothness that is adapted to the structure of the Markov process; and
 * a decomposition of random walks over the state space into multiple time scales.

I will also briefly discuss how PVFs may have applications to goal-independent learning of MDP structure, as well as potential pitfalls.  No prior knowledge of the topic will be required.

References
----------

Sridhar Mahadevan and Mauro Maggioni. "Proto-value Functions: A Laplacian Framework for Learning Representation and Control in Markov Decision Processes". Journal of Machine Learning Research, 8 (2007). pp. 2169-2231 (URL: http://jmlr.csail.mit.edu/papers/volume8/mahadevan07a/mahadevan07a.pdf)
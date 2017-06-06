Introduction
============


Correlation criterion
---------------------


The cross-correlation criterion used in this library is the zero-normalized cross-correlation (ZNCC) function [1]_

.. math::

    C_{\rm ZNCC} = \sum_{i,j=-M}^{M} \frac{ [f(x_i', y_i') - f_m] \times [g(x_i', y_i') - g_m]  }{\Delta f \Delta g}

where M is the half-width of the subset image with

.. math::

    f_m = \frac{1}{ (2M+1)^2 } \sum_{i,j=-M}^{M}  f(x_i,y_i)

and

.. math::

    \Delta f = \sqrt{ \sum_{i,j=-M}^{M}  ( f(x_i,y_i) - f_m )^2 }


The advantage of ZNCC is to be insensitive the scale and offset changes of the deformed subset intensity.

References
----------

.. [1]  Pan, B.; Qian, K.; Xie, H. & Asundi, A. Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review Measurement Science and Technology, 2009, 20, 062001 DOI:10.1088/0957-0233/20/6/062001


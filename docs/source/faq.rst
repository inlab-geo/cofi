==========================
Frequently Asked Questions
==========================

Contents
********

.. contents::
    :local:
    :class: toggle:


I have written an inverse code that uses CoFI but how do I share my example?
----------------------------------------------------------------------------

- Excellent! We use `cofi-gallery <https://github.com/inlab-geo/cofi-gallery>`_ 
  to host user-contributed CoFI showcases.
- Simply create an issue with a link to your example, a thumbnail image and a title.
  We refer you to 
  `this contributor guide <https://github.com/inlab-geo/cofi-gallery/blob/main/CONTRIBUTE.md>`_ 
  for detailed instructions.

I would like to see a specific inference method become a part of CoFI, what do I do?
------------------------------------------------------------------------------------

- If you identify a method or inference tool that suits CoFI's purpose well, you are welcome to 
  `raise an issue here <https://github.com/inlab-geo/cofi/issues>`_ or contact us directly 
  `Slack`_. 

I would like to integrate my inference method into CoFI, what do I do?
----------------------------------------------------------------------

- Please head to `CoFI's contributor guide <https://cofi.readthedocs.io/en/latest/contribute.html>`_ 
  and follow the steps under the 
  `adding a new inference tool <https://cofi.readthedocs.io/en/latest/contribute.html#new-inversion-tool>`_ 
  section.

Can I use CoFI to run my inversion in parallel if the selected tool allows that?
--------------------------------------------------------------------------------

- Yes. You can run your inversion in parallel using exactly the same ways as are 
  described in the tool's own documentation.

- Take ``emcee`` as an example, you can use CoFI to run a sampling process in parallel with
  exactly the same way as described in
  `their documentation <https://emcee.readthedocs.io/en/stable/tutorials/parallel/>`_. Click 
  `here <https://github.com/inlab-geo/cofi-examples/blob/main/examples/more_scripts/emcee_parallel_good_practice.py>`_
  for an example adapted from their documentation.

Installation Guide
==================

This document will guide you through the process of installing the ``imagematerials`` package. 

It is recommended to use a virtual environment for installing dependencies. See [this guide](https://docs.python.org/3/tutorial/venv.html) for instructions on creating and managing Python environments.

Step 1: Python Installation
---------------------------

Install Python 3.10. 


Step 2: Install PIP
----------------------

If you haven't installed pip, refer to the `Pip Installation Guide <https://pip.pypa.io/en/stable/installation/>`_ for instructions.

Step 3: Install prism
---------------------

The repository is not public, so you will need to be given access to the prism repository.

.. code-block:: console

	pip install git+https://github.com/imagepbl/prism.git

Step 4: Install pym
-------------------

.. code-block:: console

	pip install git+https://github.com/imagepbl/pym.git

Step 5: Install imagematerials
------------------------------

.. code-block:: console

	pip install git+https://github.com/imagepbl/image-materials.git

To Ensure you have the necessary dependencies installed:

.. code-block:: console

	pip install -r requirements.txt

For **pint-xarray**, install it using:

.. code-block:: console

	pip install git+https://github.com/xarray-contrib/pint-xarray

Step 6: Verifying Installation
-------------------------------

To ensure ``imagematerials`` has been successfully installed, run the following command in a Python console:

.. code-block:: python

	import imagematerials

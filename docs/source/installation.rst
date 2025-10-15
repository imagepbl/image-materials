Installation Guide
==================

This document will guide you through the process of installing the ``imagematerials`` package. 

It is recommended to use a virtual environment for installing dependencies. See [this guide](https://docs.python.org/3/tutorial/venv.html) for instructions on creating and managing Python environments.

Step 1: Install Python
---------------------------

Install Python 3.10. 


Step 2: Install PIP
----------------------

If you haven't installed pip, refer to the `Pip Installation Guide <https://pip.pypa.io/en/stable/installation/>`_ for instructions.


Step 3: Install Prerequisites
-----------------------------

General dependencies are listed in the pyproject.toml file and will be installed automatically when you install the image-materials package.
Additionally you need to install the following prerequisites:

1. pint-xarray

2. prism

3. pym

The installation can be done using pip from the command prompt or terminal. If you have python installed with anaconda, use the anaconda prompt.

Install **pint-xarray** using:

.. code-block:: console

	pip install git+https://github.com/xarray-contrib/pint-xarray

**Prism** and **pym** are not public (yet), so you will need to be given access to the GitHub repository by PBL.
Then you can install the packages by cloning the repository and installing them using pip.

.. code-block:: console

	git clone https://github.com/imagepbl/pym.git
	git clone https://github.com/imagepbl/prism.git

.. code-block:: console

	pip install ./pym
	pip install ./prism

.. or using this?:
   pip install git+https://github.com/imagepbl/prism.git
   pip install git+https://github.com/imagepbl/pym.git
   pip install git+https://github.com/imagepbl/image-materials.git


Step 4: Install IMAGE-Materials
-------------------------------

Clone the repository and install it using pip.

.. code-block:: console

	git clone https://github.com/imagepbl/image-materials.git

.. code-block:: console

	pip install image-materials

For developers:

.. code-block:: console

	pip install -e image-materials

Using -e ensures automatic updates when modifying the package.

To install additional dependencies for documentation and testing, run

.. code-block:: console

	pip install -e ".[all]"


Step 5: Verify Installation
-------------------------------

To ensure ``imagematerials`` has been successfully installed, run the following command in a Python console:

.. code-block:: python

	import imagematerials



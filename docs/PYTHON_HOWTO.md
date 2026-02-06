# HOW TO: WORK WITH PYTHON & PACKAGES
Python is one of the core languages of the IMAGE framework. For new IMAGE-related projects, Python is the recommended language due to its flexibility, wide-spread use and beginner-friendliness.

Getting started with Python can sometimes be a struggle when attempting to manage all the external packages (e.g. `numpy`, `pandas`, `scipy`) across your projects. This is where environments come in.

A Python environment is a collection of a Python version and a selection of packages, which can be activated and deactivated at will. This allows you to work on various projects on your computer in parallel, each with a separate environment. If you are confused about the concept of Python environments, please look up one of the many guides online. For the remainder of this document, it's important to have a good grasp on what they are.

To manage environments, [conda](https://docs.conda.io/projects/conda/en/stable/index.html) is IMAGE's chosen Python environment manager. Conda keeps track of which versions of Python and its packages are compatible with each other and lets you keep multiple Python environments in parallel to be used when needed.

## Quickstart
If you are working on Python projects within an IMAGE Azure desktop, in 90% of the cases you will want to use one of the shared environments found at `X:\environments`: `imagepy312_dev` as a complete environments for model development, or `imagepy312_base` for the same environment but without installable IMAGE Python packages (I.e. `prism`, `pym`. This is useful when working on `prism` or `pym` itself). See [this section](#image-conda-environments) for more information on the available shared environments. From a command line, you activate e.g. the `dev` environment as follows:
```
> conda activate imagepy312_dev
```
After this command, you can verify that you have a Python 3.12 interpreter available:
```
> python --version
Python 3.12.5
```
You can also see all the installed packages:
```
> conda list
# packages in environment at X:\environments\imagepy312_dev:
#
# Name                    Version                   Build  Channel
accessible-pygments       0.0.5              pyhd8ed1ab_0    conda-forge
aiofiles                  24.1.0             pyhd8ed1ab_0    conda-forge
...
```
When you want to use the `imagepy312_dev` environment for running Python in an IDE (e.g. Visual Studio Code), you should select the Python interpreter at `X:\environments\imagepy312_dev\python.exe` when prompted.

If the `imagepy312_dev` environment is not suitable for your work (e.g. you need Python packages not available in these environments), keep reading this guide.

## Conda
Conda is the tool for managing Python environments. It can be used to create and remove conda environments, install and uninstall packages from them, and activate and deactivate them.

What differentiates conda from e.g. using pip with Python [virtual environments](https://docs.python.org/3/library/venv.html) is that conda keeps track of dependencies between versions of packages: E.g. if you have `numpy` version `1.24.1` installed, and then you try to install a specific `pandas` version: `conda install pandas=1.4.4`, then conda will see that this version of `pandas` needs `numpy>=1.26.4`. It will then prompt you to upgrade the `numpy` installation in your environment to a version that is compatible.

Conda can be installed in different ways, through different so-called 'distributions'. Examples of conda distributions are Anaconda, Miniconda and Miniforge. All of these install conda, but with slightly different configurations. On the IMAGE Azure desktops, the current installed conda distribution is [Miniforge](https://github.com/conda-forge/miniforge).

When installing packages with conda, there are various online 'databases' where conda may look for packages. Conda calls these 'channels'. For IMAGE we strongly advise to only use the [conda-forge](https://conda-forge.org/) channel. Using multiple channels can lead to conflicts and unstable environments. The Miniforge distribution of conda by default has conda-forge set as the only channel.

For more detailed information on conda, check out [the docs](https://docs.conda.io/projects/conda/en/stable/index.html). An especially helpful resource for handling conda environments can be found on the '[Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)' page.

## IMAGE conda environments
A handful of core IMAGE conda environments have been prepared. These are managed through a [GitHub repository](https://github.com/imagepbl/conda-envs), and are available on the IMAGE Azure desktops at `X:\environments\`. These environments differ from each other in the Python version and their intended use:
* `dev`: Environments with a broad range of commonly used packages ready to be used for development of IMAGE models. They include IMAGE installable packages (`prism`, `pym`) so that you are ready to get started immediately. In most cases, the latest release of installable IMAGE packages is used. If you need to update to the latest versions of these packages (e.g. the latest `main` commit of `prism`), run `pip install git+https://github.com//imagepbl/prism.git@main`[^1].
* `base`: A duplicate of the `dev` environment, but without the IMAGE installable packages, i.e., with only the external packages like `numpy`, `pandas` etc. This can be useful as a basis of a local environment if you are developing core IMAGE packages like `prism` or `pym` and need to have a locally editable version of these packages.

[^1]: Alternatively, use the SSH url `git+ssh://git@github.com/imagepbl/prism.git@main` when authenticating to GitHub using SSH.

As for the various Python versions: the currently advised Python versions to use is 3.12. However, it is preferable that IMAGE Python projects support Python versions down to 3.11.

If you want to use one of the IMAGE conda environments but are missing a package and believe it should be part of the IMAGE environments, you may make an issue for it on the [conda-envs](https://github.com/imagepbl/conda-envs) GitHub repository. Please indicate in the issue why you need the package and why it would be useful for other IMAGE users as well.
If the request is granted, the relevant environment files in the repository will be adapted and the environment(s) on `X:\environments` will be updated.

## Azure desktops
### Shared environments
As mentioned, on Azure desktops the core IMAGE environments are available at `X:\environments`. These environments are read-only for normal IMAGE users, meaning users can't accidentally (un)install packages to it.

### Local environments
If you need to have your own environment because you want to try out some stuff or the IMAGE environments at `X:\environments` are not sufficient, you may create your own environments. Also note that performance is a reason: `X` as a shared network drive is a slow option to run Python from, so you may prefer to work using environments installed at an SSD such as `K`.
There are three main ways to create your own conda environment:
* Create a clone of another environment and then adapt it
* Create an environment based on an `environment.yml`
* Create a new environment from scratch

If you want to base your local environment on an existing environment (in this case to create a cloned environment called `imagepy312_clone` based on the `imagepy312_dev` environment), use the following command:
```
conda create --name imagepy312_clone --clone imagepy312_dev
```
To create an environment based on an environment.yml (e.g. if created by someone else for a specific project, or one of the IMAGE environments at [conda-envs](https://github.com/imagepbl/conda-envs)):
```
conda env create -f environment.yml
```
NB: the name of the environment will be taken from the first line of the `environment.yml` file. If you want to overrule that and provide your own name, provide an extra `--name myenvname` argument pair.

To create a new environment from scratch, use:
```
conda create --name myenv python=3.12
```
or whichever Python version you need for your environment.
Follow up these commands by activating the environment and installing your desired packages:
```
conda activate myenv
conda install some-obscure-package-not-in-standard-env
```
A list of all available packages in conda-forge can be found [here](https://conda-forge.org/packages/).

On the IMAGE Azure desktops, new environments are by default created at `K:\environments`. This is the recommended location to install local environments.
The `K` (data) drive is faster than the `X` (network) drive both for creating environments and for running Python from it.
Keep in mind, however, that the `K` drive is local to your desktop, and other people do not have access to it.
To share a local environment with someone else, please [export an environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment) and let the other person [reproduce the environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) on their own machine.

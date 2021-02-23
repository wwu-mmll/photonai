# Build documentation
If you want to work on the documentation, this is a short description on how to build 
it yourself using mkdocs.

## Prerequisites
In order to avoid version conflicts and problems, we recommend creating a new conda
environment and installing the PHOTONAI requirements as well as the mkdocs_requirements in this 
directory.

Additionally, you will need to install tensorflow. We will need this to build the
docs for any keras models.

## Build site
To build the documentation, cd to the project root (where the mkdocs.yaml is located) and run 
`mkdocs serve`. A documentation website will be served locally and you can open the 
documentation in your web browser.
# Economic Planning

[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://pablovegan.github.io/Economic-Planning/)
[![release](https://img.shields.io/github/v/release/pablovegan/python-tips-tools.svg)](https://github.com/pablovegan/Python-tips-tools/releases/latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Test](https://github.com/pablovegan/Python-tips-tools/actions/workflows/tests.yml/badge.svg)](https://github.com/pablovegan/Python-tips-tools/actions/workflows/tests.yml)

A simple example package for economic planning.

## Documentation and examples
Documentation and examples can be found in https://pablovegan.github.io/Economic-Planning/.

## Installation

With `pip`:
```console
pip install 
```

## Quick usage

_Work in progress..._

## To do
[ ] Add pollutant plots to Spain's notebook


## Understanding the algorithm

### Autarky

Let's begin with a simple example: a self-sufficient economy without external commerce. In order to feed our population, we need to produce more than we consume:
$$supply \geq final_{domestic} $$
But something is wrong. In order to produce, we also need to consume goods, so we need to substract the used goods in the production:
$$supply \geq use_{domestic} + final_{domestic} $$
This is going to be our main constraint. From now on, we are simply going to add more complexity to the model in order to make it more realistic.

Since we are producing more than we need, there will possibly be some excess production that we can reuse in the next period. But not all excess will be available: food spoils, machines wear out, etc. To model this, we add a depreciation
$$depreciation * excess + supply \geq use_{domestic} + final_{domestic}$$

### Introducing trade

What happens if we need to import part of the goods from external economies, both for final consumption and for intermediate production? 
$$supply + import \geq use_{domestic} + use_{import} + final_{domestic} + final_{import}$$
(To avoid clogging the equation, we will temporarily remove the $excess$ term of the inequality).

Supposing we use in each period what we import, we can equal
$$import \approx use_{import} + final_{import}$$
and remove the terms from the inequality (this restriction can be relaxed). 

This may seem like imports are not considered in our algorithm... but they are. We cannot import all the goods we want, if that happened we wouldn't need to work! To import goods from other economies we need to export something in exchange.
$$excess + supply \geq use_{domestic} + final_{domestic} + final_{export}$$
This allows us to introduce yet another kind of restriction: a balance of trade. To do this we need some kind of unifying measure to compare 'how much' we import and export. Usually, this is done with one-dimensional quantity, prices, but we could use other measures like labour time, energy or even multidimensional units. For now, lets keep it simple and assume we want to export more than we import:
$$price_{export} \cdot final_{export} \geq price_{import} \cdot (use_{import} + final_{import})$$


## References

This package is inspired by Hagberg's and Zacharia's receding horizon planning [repository](https://github.com/lokehagberg/rhp) and Hagberg's [thesis](https://www.diva-portal.org/smash/get/diva2:1730354/FULLTEXT01.pdf).

Great Python packages exist for input-output analysis, such as [Pymrio](https://github.com/IndEcol/pymrio), altough it is not focused on planning.

_Pymrio – A Python Based Multi-Regional Input-Output Analysis Toolbox._ Konstantin Stadler (2021).
https://openresearchsoftware.metajnl.com/articles/10.5334/jors.251

Life cycle assessment package https://github.com/brightway-lca/brightway2

ODYM - Open Dynamic Material Systems Model https://github.com/IndEcol/ODYM

Lifting Industrial Ecology Modeling to a New Level of Quality and Transparency: A Call for More Transparent Publications and a Collaborative Open Source Software Framework https://onlinelibrary.wiley.com/doi/full/10.1111/jiec.12316

J. M. Montias. [Planning with Material Balances in Soviet-Type Economies](https://sci-hub.st/https://www.jstor.org/stable/1813077)

Spanish supply-use tables can be found in the website of the [National Statistics Institute](https://www.ine.es/dyngs/INEbase/en/operacion.htm?c=Estadistica_C&cid=1254736177059&menu=resultados&idp=1254735576581) (INE).

Introduction to control theory. https://fab.cba.mit.edu/classes/865.21/topics/control/


## Contributing

We appreciate and welcome contributions. For major changes, please open an issue first
to discuss what you would like to change. Also, make sure to update tests as appropriate.

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.

## License

This software is under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

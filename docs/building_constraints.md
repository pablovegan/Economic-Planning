# Receding Horizon planning

Remember that we already defined two basic ingredients of our linear programming problem:
1. We want to minimize the monetary cost of goods produced.
2. The variables are the activity level of each unit of production.

## Autarquía

Partimos primero de una economía autárquica. La restricción principal a la producción es que debemos dar más de lo que la gente va a consumir:
$$supply \geq target\_national$$
Pero algo falla, para producir necesitamos consumir unos bienes de producción. Lo arreglamos:
$$supply - use \geq target\_national$$
Si además tenemos en cuenta stock sobrante del periodo pasado, obtenemos
$$excess + supply - use \geq target\_national$$
Bajo una hipótesis más realista, parte del exceso del periodo anterior se perdería, por lo que habría que añadir una depreciación:
$$depreciation * excess + supply - use \geq target\_national$$


## Economía dependiente

¿Qué sucede ahora si debemos importar parte de lo que usa nuestra economía nacional para producir?
$$
\begin{split}excess + supply - use + import\_industry - use\_import \\
\geq target\_national
\end{split}
$$
De momento podemos asumir
$$import\_industry \approx use\_import$$
También podemos añadir un target de exportaciones
$$excess + supply - use \geq target\_national + target\_export$$
e incluso un vector de importaciones para consumo final
$$\begin{split} excess + supply - use + import\_consum \\
\geq target\_national + target\_export \end{split}$$

En caso de que pudiera resultar menos costoso acumular productos importados para su uso más adelante (porque los precios estuviesen bajos), tendríamos que añadir de nuevo la matriz $use\_import$:
$$\begin{split} excess + supply - use - use\_import + import\_total \\
\geq target\_national + target\_export \end{split}$$
siendo 
$$import\_total = import\_industry + import\_consum$$


## Ideas

Probar Robust optimization
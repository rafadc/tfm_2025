# Mejorando el rendimiento de benchmarks usando promptbench

El proposito de este experimento es comprobar si se mejora el rendimiento de el benchmark MMLU para evaluar el rendimiento de los distintos LLM si hacemos una primera fase de mejora de prompt para el LLM especifico utilizando el resultado de promptbench

El proceso es como sigue
- Utilizamos el benchmark MMLU como base
- Para cada una de las LLM a probar la utilizamos a si misma para generar 50 variantes del prompt que se va a utilizar y utilizamos promptbench para evaluar cual es el mejor
- Sustituimos el prompt en el benchmark para esa LLM por el que mejor resultado ha obtenido
- Volvemos a ejecutar el benchmark y comparamos con la linea base

Tendremos una lista de llms sobre las que vamos a probar el metodo. En el codigo se proveen ejemplos usando 'gpt-oss:latest' y 'llama3.2:latest'.

## Paso 1: Descarga de datos de benchmark MMLU

Descargar los datos del benchmark. 
Si la carpeta ```data``` no existe descargar el fichero https://people.eecs.berkeley.edu/~hendrycks/data.tar y lo extraera. Deberia crear la carpeta ```data```. Si ya existe la carpeta no hay nada que hacer. 
Estos son los datos del benchmark MMLU.

Ejecutaremos el benchmark MMLU con la implementacion de HuggingFace con los datos descargados y las LLM a probar. Almacenaremos el resultado en fichero. Si el fichero de resultado existe previamente no lo regeneraremos para ahorrar tiempo.

## Paso 2: Generacion de alternativas de prompt

Crearemos una carpeta para cada llm con el nombre data\_nombrellm. Por ejemplo data\_gpt-oss o data\llama3.2.

En esa carpeta copiaremos la carpeta ```test``` original de data.

En ```data/test``` hay una serie de ficheros csv que corresponden a preguntas de test. En el codigo utilizaremos solamente ```anatomy_test.csv```.

El formato de esos ficheros es

```csv
A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral,paralysis of the facial muscles.,paralysis of the facial muscles and loss of taste.,"paralysis of the facial muscles, loss of taste and lacrimation.","paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.",A
"A ""dished face"" profile is often associated with",a protruding mandible due to reactivation of the condylar cartilage by acromegaly.,a recessive maxilla due to failure of elongation of the cranial base.,an enlarged frontal bone due to hydrocephaly.,defective development of the maxillary air sinus.,B
Which of the following best describes the structure that collects urine in the body?,Bladder,Kidney,Ureter,Urethra,A
Which of the following structures is derived from ectomesenchyme?,Motor neurons,Skeletal muscles,Melanocytes,Sweat glands,C
Which of the following describes the cluster of blood capillaries found in each nephron in the kidney?,Afferent arteriole,Glomerulus,Loop of Henle,Renal pelvis,B
A patient suffers a broken neck with damage to the spinal cord at the level of the sixth cervical vertebra.,They will be unable to breathe without life support.,They will only be able to breathe quietly.,It is impossible to predict an effect on breathing.,Breathing will be unaffected.,B
```

Extraeremos el primer campo que corresponde la pregunta y crearemos un fichero analogo que se llamara anatomy\_test\_prompts.csv en cada linea tendremos 10 prompts separados por comas que corresponden a esa linea en el test.

Para generar el nuevo prompt preguntaremos a la misma LLM que estamos probando.

No generaremos el fichero de nuevo si ya esta creado

## Paso 3: Evaluacion de los prompts usando promptgen

Utilizando promptbench evaluaremos los prompts del fichero que hemos generado en el paso anterior y sustituiremos el prompt en el fichero del test por el mejor.

## Paso 4: Ejecucion de MMLU con los nuevos prompts

Ejecutaremos otra vez el benchmark MMLU pero esta vez utilizando los nuevos ficheros que hemos generado en el Paso 3

## Paso 5: Graficar los resultados

Haremos una grafica y una tabla comparando los resultados con los prompts originales y los prompt que resultan mejores usando promptbench

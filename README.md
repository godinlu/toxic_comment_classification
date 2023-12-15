# toxic_comment_classification
On implémente ici un CountVectorizer qui sert à transformer un texte en vecteur avec un principe de comptage.
Par exemple si l'on donne la phrase "bonjour je m'appelle didier" alors la phrase sera transformer en
"bonjour","je", "appelle", "didier"
1           1       1          1

Et une fois tous les mots du data_train appris seul les mots présent dans le data_train pourront être enregistré
Pour continuer notre exemple si l'on injete la phrase "bonjour je suis luc" alors la phrase sera transformer en
"bonjour","je", "appelle", "didier"
1           1       0          0
Il est donc important d'avoir beaucoup de mot dans les données d'entrainement car les nouveau mots ne seront pas pris en compte.






ensuite la fonction `TfidfVectorizer` qui vient de Term "Frequency-Inverse Document Frequency Vectorizer"
qui marche globalement sur le même principe que `CountVectorizer` mais cela donne un résultat plus précis, 
avec la fréquence et l'importance des mots.

On observe alors que dans nos données d'entrainements, on compte en tout 189775 mots différents. Il est important
de notifier que tous mots qui ne sont pas dans cette liste seront inconnue. 

Ici on créer un réseau de neuronne avec 1 couche caché de 64 neuronnes. On choisit de faire seulement 10 itération car on a une grande quantité de données donc le modèle aprend beaucoup à chaque itération. On a choisit la fonction d'activation 'logistic' car c'est la plus efficace dans notre cas.
On observe cependant que le modèle est un peu moin bien que la régression logistique. On peut expliquer cela car nous avons choisit de mettre peu de neuronnes et peut de couches. Or il aurait surement été plus efficace d'augmenter cela, mais le temps de calcule devients très vite long. 
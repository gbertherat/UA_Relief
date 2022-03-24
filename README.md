# Relief


Pour lancer le projet:<br>
``java -jar Relief.jar {nom_fichier} [m] [k]``

Version de java: 1.8

Arguments:<br>
{} = Obligatoire<br>
[] = Optionel

- ``{nom_fichier}`` : Le nom du fichier .arff à utiliser entre **heart_statlog**, **iris2Classes** ou **optdigits_39**.
- ``[m]`` : Le nombre d'itérations `m` pour l'algorithme Relief.
- ``[k]`` : Le nombre de voisins `k` à récupérer pour déterminer les barycentres.

Si ``[m]`` n'est pas spécifié, m = nombre d'instances.<br>
Si ``[k]`` n'est pas spécifié, k = 5.

Note: ``[m]`` doit être spécifié pour que ``[k]`` puisse être spécifié.

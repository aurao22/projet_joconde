#!/bin/bash

file_name='base-joconde-extrait_encoding.csv'

sed -i 's/Ã©/é/g' $file_name
sed -i 's/ÃŽ/à/g' $file_name
sed -i 's/ÃŽ/à/g' $file_name
sed -i "s/â€™/'/g" $file_name
sed -i "s/â€™/'/g" $file_name
sed -i "s/â€™/'/g" $file_name
sed -i "s/â€™/'/g" $file_name
sed -i "s/Ã§/ç/g" $file_name
sed -i "s/Ã§/ç/g" $file_name
sed -i "s/Ã /à/g" $file_name
sed -i "s/Ã¢/â/g" $file_name
sed -i "s/Ã´/ô/g" $file_name
sed -i "s/Ã  /à /g" $file_name
sed -i "s/Â°/°/g" $file_name
sed -i "s/Ã®/î/g" $file_name
sed -i "s/Ãª/ê/g" $file_name
sed -i "s/Â«/«/g" $file_name
sed -i "s/Â»/»/g" $file_name
sed -i "s/Ã¹/ù/g" $file_name
sed -i "s/Â©Â/©/g" $file_name
sed -i "s/àle-de-France/Ile-de-France/g" $file_name

# Récupération des noms d'artiste contenu dans les commentaires
# grep "'e : .*'" name.txt
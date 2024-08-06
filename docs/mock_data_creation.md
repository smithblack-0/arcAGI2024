# Introduction

We will need mock data

## Outline

* Define data creator
  * White noise adequete - > convert into grid
* Define transforms on data
  * Selection mode
    * Region
    * Color
  * Transforms
    * Recolor
    * Translate
    * Reflect
    * Tile
    * Repeat in dir
  * Combine with original
* Planning and Scheduling
  * Based on whether or not the model is having trouble.
  * Use simple markov model to control and update probabilities based on previous n-gram of actions.
  * Increase number of transforms applied as model becomes more capable.
  * Recycle particularly tricky challenges


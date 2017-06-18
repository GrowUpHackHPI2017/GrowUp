"# GrowUp" 
## Inspiration
Continous heigth tracking of children is crucial to monitoring a healthy child development. Moreover, height tracking is an easy and effortless task, every parent can do on their own. Theirfore we decided to create an application to simplify the process of height tracking.

## What it does
We use a door frame as a reference frame for analysing the child's height from a simple image. The door frame's height is known and is used as a reference distance. We calculate the child's height in relation to that.

## How we built it
We used Python2 and mainly build on openCV to process the images. Additionally, we created a GUI as a first implementation of our approach.

## Challenges we ran into
It is surprisingly hard to track a door frame; more sophisticated algorithms can greatly improve its current performance

## Accomplishments that we're proud of
We got a first draft of the application to work and were able to visualize our ideas and put them in a mock-up for further development.

## What we learned
Available API cloud solutions might be easier to use, but do not necessarily provide all the options you might need. We deep dived into image processing, learning a lot about object recognition in openCV.

## What's next for GrowUp
Improving the height estimate in a two-fold manner: On the one hand improving the performance of door frame recognition, and on the other hand better human tracking. Furthermore, additional features, such as face recognition and personal moment should be implemented. Along side, our GUI should be transformed to an actual mobile app.

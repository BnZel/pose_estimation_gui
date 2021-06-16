# Experimenting with Pose Estimation

## Overview
With an interest towards Computer Vision, I also was fascinated by the idea of pose estimation and wanted to utilize it into a User Interface

## Resources
***NOTE: Unfortunately the pose estimation repository from [ildoonet](https://github.com/ildoonet) has been removed, I had to alot of searching for other alternatives which I cannot recall.***
* [Mark Jay's Pose Estimation Tutorial](https://www.youtube.com/playlist?list=PLX-LrBk6h3wQ17z1axCOAS1QVS1dvTEvR)
* PyQT5 and PyQTGraph
* OpenCV

## Project In Detail 

Defined **connections** (found within the pose estimation repo) that would be later added to the plot as well as **lines**
 ```python
        # in initUI()
        self.lines = {}
        self.connection = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]
```        
### Layout
```python
 # ----------left layout----------
        self.layoutLeft = QGridLayout()

        # input display
        self.inputDisplay = QWidget()
        self.width = 740
        self.height = 580

        self.image_label = QLabel(self)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image_label)
        self.inputDisplay.setLayout(self.vbox)

        grey = QPixmap(self.width, self.height)
        grey.fill(QColor('darkGray'))
        self.image_label.setPixmap(grey)

        # buttons
        self.leftButtons = QWidget()
        self.btnUpload = QtWidgets.QPushButton(self)
        self.btnUpload.setText("Upload")
        self.btnUpload.clicked.connect(self.uploadFile)
        self.btnUpload.setDisabled(False)

        self.btnClear = QtWidgets.QPushButton(self)
        self.btnClear.setText("Clear")
        self.btnClear.clicked.connect(self.clearInput)
        self.btnClear.setDisabled(True)

        self.layoutLeft.addWidget(self.btnUpload,1,0)
        self.layoutLeft.addWidget(self.btnClear,1,1)

        self.layoutLeft.setRowStretch(4,1)
        self.leftButtons.setLayout(self.layoutLeft)
```
![Left Layout](/demo/UI-left.PNG "Left Layout")

```python
        # ----------right layout----------
        self.outputDisplay = QWidget()
        width = 740
        height = 580

        # where the pose estimation would display
        self.window = gl.GLViewWidget()
        self.window.setCameraPosition(distance=47, elevation=12)

        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()
        gx.rotate(90,0,1,0)
        gy.rotate(90,1,0,0)
        gx.translate(-10,0,0)
        gy.translate(0,-10,0)
        gz.translate(0,0,-10)
        self.window.addItem(gx)
        self.window.addItem(gy)
        self.window.addItem(gz)

        # pose estimation model
        # take keypoints from pose 
        # plot those points within output
        model = 'mobilenet_thin'
        self.w, self.h = model_wh('432x368')
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w,self.h))
        self.poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')

        self.layoutRight = QVBoxLayout(self)
        self.layoutRight.addWidget(self.window)
        self.outputDisplay.setLayout(self.layoutRight)

        # keypoints table
        self.lblKeypoints = QLabel(self)
        self.lblKeypoints.setFont(QFont('Bold',15))
        self.lblKeypoints.setText("Keypoints")
        self.tableKeypoints = QTableWidget(self)
        self.tableKeypoints.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.tableKeypoints.resize(440,550)

        # connections table
        self.lblConnections = QLabel(self)
        self.lblConnections.setFont(QFont('Bold',15))
        self.lblConnections.setText("Connections")
        self.tableConnections = QTableWidget(self)
        self.tableConnections.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.tableConnections.resize(440,550)

        self.rightItems = QWidget(self)
        self.layoutRightItems = QGridLayout(self)
        self.layoutRightItems.addWidget(self.lblKeypoints,6,0)
        self.layoutRightItems.addWidget(self.tableKeypoints,7,0)
        self.layoutRightItems.addWidget(self.lblConnections,6,1)
        self.layoutRightItems.addWidget(self.tableConnections,7,1)
        self.rightItems.setLayout(self.layoutRightItems)
```
![Right Layout](/demo/UI-right.PNG "Right Layout")

### Upload 
```python
    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Upload Image", "","PNG (*.PNG);; JPEG (*.JPEG) (*.JPG)", options=options)

        return self.convert_cv_qt(fileName)
        
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""

        if Path(cv_img).suffix == ".PNG".casefold() or (Path(cv_img).suffix == ".JPEG".casefold() or Path(cv_img).suffix == ".JPG".casefold()):
            img = common.read_imgfile(cv_img)

            qimage = self.Worker.processImage(img, self.width, self.height)   # Worker Class (Thread) function
            self.plotPose(img)

            self.btnUpload.setDisabled(True)

            return self.image_label.setPixmap(QPixmap.fromImage(qimage))  
```

### Display

```python
    def plotPose(self, filename):
        self.kp = self.mesh(filename)

        print("keypoints shape: ",np.shape(self.kp))

        self.points = gl.GLScatterPlotItem(
            pos = self.kp,
            color = pg.glColor((0,255,0)),
            size = 15
        )
        self.window.addItem(self.points)   

        self.saveKeypoints()                  # dictionary updates whenever each pose keypoints are plotted
                    
        # each line into dictionary
        # for every point get two numbers within connection dictionary
        for n, pts in enumerate(self.connection):
            self.lines[n] = gl.GLLinePlotItem(
                pos = np.array([self.kp[p] for p in pts]),
                color = pg.glColor((0,0,255)),
                width = 3,
                antialias=True
            )
            self.window.addItem(self.lines[n]) 

        self.saveConnections(self.lines)      # dictionary updates whenever each pose connections are plotted

        self.btnClear.setDisabled(False)
```

The **dimensions of a given image would be 3**, however I could not work around plotting things less than 3 with spotted remaining **lines** and **connections**
```python
    def mesh(self, image):

        image_h, image_w = image.shape[:2]
        width = 640
        height = 480 
        pose_2d_mpiis = []
        visibilities = []
        humans = self.e.inference(image, resize_to_default=False, upsample_size=4.0)
        img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append([(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii])
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        
        print("number of dimensions: ",pose_2d_mpiis.ndim)    # accepts 3 dimensions, otherwise would give an assertion error

        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)

        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        keypoints = pose_3d[0].transpose()          # turn to numpy list and send the data to plot the pose within the display

        print(keypoints)

        # large coords will extend outside frame
        # divide to display within frame
        return keypoints / 80 
```

### Displaying Keypoints and Connections Data

Issue regarding selecting a specific row to plot (as it plots the latest one)

Dictionary would be cleared as user exits out the program
```python
 def saveKeypoints(self):
        self.save_keypoints.update({self.k:self.kp})
        self.k += 1

        # update to table
        row_count = (len(self.save_keypoints))
        col_count = (len(self.save_keypoints[0]))

        print("row length: ",row_count,", column length: ",col_count)

        self.tableKeypoints.setRowCount(row_count)
        self.tableKeypoints.setColumnCount(col_count+1)

        for row in range(row_count):
            for col in range(col_count):
                item = (list(self.save_keypoints[row])[col])          # update to keypoints dictionary

                self.replotKeypoints = self.save_keypoints[row]

                self.btnPlot = QtWidgets.QPushButton("Plot", self)
                self.btnPlot.clicked.connect(self.replotPose)
                self.btnPlot.focusWidget()

                self.tableKeypoints.setItem(row,col,QTableWidgetItem(str(item)))
                self.tableKeypoints.setCellWidget(row,17,self.btnPlot)            # adds extra row for plot button to be added (counting from the length of rows and columns lists the pose has)
                index = self.tableKeypoints.indexAt(self.btnPlot.pos())
        
    def saveConnections(self, lines):        
        self.save_connections.update({self.c:lines[self.c].pos})      # update to connections dictionary
        self.c += 1

        row_count = (len(self.save_connections))
        col_count = (len(self.save_connections[0]))

        self.tableConnections.setRowCount(row_count)
        self.tableConnections.setColumnCount(col_count)

        for row in range(row_count):
            for col in range(col_count):
                item = (list(self.save_keypoints[row])[col])
                self.tableConnections.setItem(row,col,QTableWidgetItem(str(item)))

        print(self.save_connections.items())
```
![Keypoints and Connections Table](/demo/Keypoints-and-Connections-Table.PNG "Keypoints and Connections Table")

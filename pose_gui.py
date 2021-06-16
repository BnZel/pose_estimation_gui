from __init__ import *

class Worker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def processImage(self, img, width, height):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return p

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(10,50, 1980, 1080)
        self.setWindowTitle('UI')
        self.initUI()

    def initUI(self):
        self.mainLayout = QGridLayout()
        self.Worker = Worker()
        self.imgCopy = ''

        self.save_keypoints = defaultdict(list)
        self.save_connections = defaultdict(list)
        self.k = 0
        self.c = 0

        self.lines = {}
        self.connection = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

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

        # ----------right layout----------
        self.outputDisplay = QWidget()
        width = 740
        height = 580

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

        # add to main layout
        self.mainLayout.addWidget(self.inputDisplay,0,0)
        self.mainLayout.addWidget(self.leftButtons,1,0)
        self.mainLayout.addWidget(self.outputDisplay,0,1)
        self.mainLayout.addWidget(self.rightItems,1,1)
        self.setLayout(self.mainLayout) 

    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Upload Image", "","PNG (*.PNG);; JPEG (*.JPEG) (*.JPG)", options=options)

        return self.convert_cv_qt(fileName)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""

        if Path(cv_img).suffix == ".PNG".casefold() or (Path(cv_img).suffix == ".JPEG".casefold() or Path(cv_img).suffix == ".JPG".casefold()):
            img = common.read_imgfile(cv_img)

            qimage = self.Worker.processImage(img, self.width, self.height)
            self.plotPose(img)

            self.btnUpload.setDisabled(True)

            return self.image_label.setPixmap(QPixmap.fromImage(qimage))         
                
    def clearInput(self):
        print("clear input")
        grey = QPixmap(self.width, self.height)
        grey.fill(QColor('darkGray'))

        lbl = self.image_label.setPixmap(grey)
        pts = self.window.removeItem(self.points)
        lns = self.lines

        for i in list(self.lines):
            lns = self.window.removeItem(self.lines[i])

        self.btnClear.setDisabled(True)
        self.btnUpload.setDisabled(False)

        return lbl, pts, lns

    def plotPose(self, filename):
        self.kp = self.mesh(filename)

        print("keypoints shape: ",np.shape(self.kp))

        self.points = gl.GLScatterPlotItem(
            pos = self.kp,
            color = pg.glColor((0,255,0)),
            size = 15
        )
        self.window.addItem(self.points)   

        self.saveKeypoints()
                    
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

        self.saveConnections(self.lines)

        self.btnClear.setDisabled(False)
    
    # take keypoints of pose
    # render to output display
    # compute 2d then 3d points
    # transpose points as appropriate list format
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
        
        print("number of dimensions: ",pose_2d_mpiis.ndim)

        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)

        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        keypoints = pose_3d[0].transpose()

        print(keypoints)

        # large coords will extend outside frame
        # divide to display within frame
        return keypoints / 80 

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
                item = (list(self.save_keypoints[row])[col])

                self.replotKeypoints = self.save_keypoints[row]

                self.btnPlot = QtWidgets.QPushButton("Plot", self)
                self.btnPlot.clicked.connect(self.replotPose)
                self.btnPlot.focusWidget()

                self.tableKeypoints.setItem(row,col,QTableWidgetItem(str(item)))
                self.tableKeypoints.setCellWidget(row,17,self.btnPlot)
                index = self.tableKeypoints.indexAt(self.btnPlot.pos())
        
    def saveConnections(self, lines):        
        self.save_connections.update({self.c:lines[self.c].pos})
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

    def replotPose(self):

        keypoints = self.replotKeypoints

        self.points = gl.GLScatterPlotItem(
            pos = keypoints,
            color = pg.glColor((0,255,0)),
            size = 15
        )
        self.window.addItem(self.points) 
        
        for n, pts in enumerate(self.connection):
            self.lines[n] = gl.GLLinePlotItem(
                pos = np.array([keypoints[p] for p in pts]),
                color = pg.glColor((0,0,255)),
                width = 3,
                antialias=True
            )
            self.window.addItem(self.lines[n]) 

        self.btnClear.setDisabled(False)

    def onSelectionChanged(self, selected, deselected):

        # get cells location
        for ix in selected.indexes():
            print('selected at: ROW: {0}, COLUMN: {1}'.format(ix.row(), ix.column()))
            print('CURRENT ITEM: ',self.tableKeypoints.currentItem().text())
            keypoints = self.tableKeypoints.item(self.tableKeypoints.currentRow(), ix.column()).text()
            print('CURRENT ITEM: ',keypoints)

        for ix in deselected.indexes():
            print('deselect at: ROW: {0}, COLUMN: {1}'.format(ix.row(), ix.column()))

if __name__ == "__main__":
    app = QApplication([])
    win = Window()
    win.show()
    sys.exit(app.exec_())
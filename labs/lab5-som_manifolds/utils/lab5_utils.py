import matplotlib.pyplot as plt
import numpy as np

class KohonenNetPlotter(object):
    def __init__(self, imx, imy, W, numTrials=0):
        self._imx = imx
        self._imy = imy
        self._W = W
        self._im = self.__computeRfsIm()
        self._numTrials = numTrials
        self.__initPlot()
        
        
    def __initPlot(self):
        figure, axes = plt.subplots(1,2)
        self._mapAxes  = axes[0]
        self._mapAxes.set_title("Topological Map")
        self._rfsAxes = axes[1]
        self._rfsAxes.axis('off')
        self._rfsAxes.set_title("Receptive Fields")
        self._figure = figure
        self._figure.suptitle("Kohonen Net Result at step: %d" % self._numTrials)
        plt.tight_layout()
        self.updatePlot(self._W)
    
    def __computeMap(self):
        imx = self._imx
        imy = self._imy
        W = self._W
        mux=np.abs(np.dot(W.T, imx.ravel())/np.sum(np.abs(W),0).T)
        mux.resize(imx.shape)
        muy=np.abs(np.dot(W.T, imy.ravel())/np.sum(np.abs(W),0).T)
        muy.resize(imx.shape)
        return mux,muy
        
    def __computeRfsIm(self,bg='min',im=None):
        """
        Parameters
        ----------
        A : array
            Receptive field / basis function array of shape L,M where L is a
            square number which corresponds to the number of pixels, and M is the
            number of basis functions (either square or some power of 2)
        bg : 'min' or 'max'
            Set the 'background' (border between RFs) to be either the maximum or
            the minimum value of the current colormap

        im : AxesImage
            will use im._A as the array into which the basis functions will be
            arranged

        Returns 
        -------
        im : AxesImage 
            created by the plt.imshow() commandwhich contains the RFs / basis
        functions (so you can update them)

        Notes
        -----
        This is Bruno Olshausen's ``showrfs.m`` as pythonized by Paul Ivanov
        Further pythonized by Yubei Chen
        """
        W = self._W
        L,M=W.shape;
        sz=np.int(np.sqrt(L));

        if np.floor(np.sqrt(M))**2 != M:
            m=np.int(np.sqrt(M/2))
            n = M/m
        else:
            n=m=int(np.sqrt(M))

        buf=1; # border around RFs

        # allocate one array that all of the basis functions will be inserted into
        if im is None:
            ar=np.ones((buf+m*(sz+buf),buf+n*(sz+buf)));
            if bg=='min':
                ar*=-1
        else:
            ar = im

        k=0;

        for j in range(m):
            for i in range(n):
                clim=np.max(np.abs(W[:,k])); # rescale basis function for display
                x0,y0 = buf+j*(sz+buf),buf+i*(sz+buf)   # offset for basis function
                sl = np.index_exp[x0:x0+sz,y0:y0+sz]    # slice into array
                ar[sl] =  W[:,k].reshape(sz,sz)/clim;
                k+=1
        return ar
        
    
    def updatePlot(self,W,numTrials=0):
        self._W = W
        self._numTrials = numTrials
        
        #Update Map
        mux,muy = self.__computeMap()
        self._mapAxes.plot(mux,muy,'k')
        self._mapAxes.hold(True)
        self._mapAxes.plot(mux.T,muy.T,'k')
        self._mapAxes.hold(False)
        self._mapAxes.axis([0,self._imx.shape[0],0,self._imx.shape[1]])
        self._mapAxes.set_title("Topological Map")
        self._mapAxes.set_aspect('equal')
        self._figure.suptitle("KohonenNet Result at step: %d" % self._numTrials)
        
        #Update Im:
        self._im = self.__computeRfsIm(im=self._im)
        self._rfsAxes.imshow(self._im, vmin=-1,vmax=1)
        #self._rfsAxes.axis('off')
        self._figure.canvas.draw()

class LLE_util(object):
    def __init__(self):
        pass
        
    def faceReshape(self, faceColumn):
        """
        Reshapes a column-vector yaleface into a matrix yaleface
        suitable for viewing as an image w facePlot
    
        Parameters
        ----------
        faceColumn : numpy array, column vector from yalefaces
        
        Returns
        -------
        faceImage  : numpy array, reshaped faceColumn
        """
        faceImage = np.reshape(faceColumn,(61,-1),order='C')
        return faceImage 
    
    def plotImgSamples(self, data):
        buff = 2
        imgSz = np.int(np.sqrt(data.shape[0]))
        imgNum = data.shape[1]
        dimSz = np.int(np.sqrt(imgNum))
        im = 200*np.ones([(imgSz+buff)*dimSz-buff,(imgSz+buff)*dimSz-buff])
        for i in range(dimSz):
            for j in range(dimSz):
                x0,y0 = j*(imgSz+buff),i*(imgSz+buff)
                sl = np.index_exp[x0:x0+imgSz,y0:y0+imgSz]
                im[sl] = data[:,i*dimSz+j].reshape(imgSz, imgSz)
        figure = plt.figure()
        figure.suptitle("Image Data Visualization")
        plt.imshow(im,cmap='Greys_r')
        figure.axes[0].axis('off')
        
    def generate_data(self, faces):
        
        # Select A Face Patch
        faceColumn = faces[:,3]
        faceImage = self.faceReshape(faceColumn)
        faceImage = faceImage[np.index_exp[15:54,15:46]]
        # Generate a random noise image
        randomImg = 100*np.random.rand(100,100)
        
        dimX = 30
        dimY = 34
        dataNum = dimX * dimY
        imgSz = 100
        data = np.zeros([imgSz**2,dataNum])
        translatedImg = np.zeros([imgSz,imgSz])
        
        # Get a random permutation of the index
        permute = np.random.permutation(dataNum)
        for i in range(dimX):
            for j in range(dimY):
                posx = i*2
                posy = j*2
                translatedImg[:,:] = randomImg.copy()
                translatedImg[np.index_exp[posx:posx+39,posy:posy+31]] = faceImage
                indx = permute[i*dimY+j]
                data[:,indx:indx+1] = translatedImg.reshape(imgSz**2,1)

        return data
    
    def compute_pairwise_distance(self, data):
        dimSz = data.shape[1]
        dis = data.T @ data
        diag = np.expand_dims(np.diag(dis),2)
        dis = -2*dis + np.ones([dimSz,1]) @ diag.T + diag @ np.ones([1,dimSz])
        dis = np.sqrt(dis)
        return dis
    
    def LLE_2D_plot(self, manifold):
        figure = plt.figure()
        plt.plot(manifold[0,:],manifold[1,:],'.')
        figure.suptitle("2D LLE Manifold Embedding Restult")
        figure.axes[0].axis('off')
        
        
        
        
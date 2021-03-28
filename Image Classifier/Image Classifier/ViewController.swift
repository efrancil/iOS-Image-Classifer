//
//  ViewController.swift
//  Image Classifier
//
//  Created by Ethan Franciliso on 3/26/21.
//

import UIKit
import AVKit
import Vision

class ViewController: UIViewController {
    
    @IBOutlet weak var classificationLabel: UILabel!
    var captureSession: AVCaptureSession = AVCaptureSession()
    var dataOutput: AVCaptureVideoDataOutput = AVCaptureVideoDataOutput()
    var model: VNCoreMLModel?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        initCamera()
    }
    
    // Set up the camera for live video feed
    func initCamera() {
        // photo preset gives border on top and bottom
        self.captureSession.sessionPreset = .photo
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            return
        }
        
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else {
            return
        }
        self.captureSession.addInput(input)
        
        self.captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        self.dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        self.captureSession.addOutput(self.dataOutput)
    }
    
    // Set up vision with the CoreML model
    func initVisionCoreML() -> VNCoreMLRequest {
        do {
            self.model = try VNCoreMLModel(for: CIFAR10_First_CNN(configuration: .init()).model)
        }
        catch {
            print("Error initializing CoreMLModel " + error.localizedDescription)
        }
        
        let request = VNCoreMLRequest(model: self.model!, completionHandler: { [weak self] request, error in
            self?.processClassifications(for: request, error: error)
        })
        // Use to preprocess correctly
        request.imageCropAndScaleOption = .centerCrop
        
        return request
    }
    
    // Process the classifications and update UI
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            // Handle error and get observations if good
            guard let results = request.results else {
                self.classificationLabel.text = "Unable to classify image " + error!.localizedDescription
                return
            }
            
            let classifications = results as! [VNClassificationObservation]
            let bestClassification = classifications.first
            
            // Get the first observation and update the labels
            print(bestClassification!.identifier, bestClassification!.confidence)
            DispatchQueue.main.async {
                let confidencePercent = String(format: "%.2f", bestClassification!.confidence * 100)
                self.classificationLabel.text = bestClassification!.identifier + ": " + confidencePercent + "%"
            }
        }
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Get a frame
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let request = initVisionCoreML()
        // Pass in the frame and the request

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        do {
            try handler.perform([request])
        }
        catch {
            print("Error in classification " + error.localizedDescription)
        }
    }
}


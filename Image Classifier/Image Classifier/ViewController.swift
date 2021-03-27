//
//  ViewController.swift
//  Image Classifier
//
//  Created by Ethan Franciliso on 3/26/21.
//

import UIKit
import AVKit
import Vision

class ViewController: UIViewController
{

    @IBOutlet weak var classificationLabel: UILabel!
    var captureSession: AVCaptureSession = AVCaptureSession()
    var dataOutput: AVCaptureVideoDataOutput = AVCaptureVideoDataOutput()
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        setUpCamera()
    }
    
    func setUpCamera()
    {
        self.captureSession.sessionPreset = .photo
        guard let captureDevice = AVCaptureDevice.default(for: .video) else
        {
            return
        }
        
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else
        {
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
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate
{
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection)
    {
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else
        {
            return
        }

        guard let model = try? VNCoreMLModel(for: CIFAR10_First_CNN(configuration: .init()).model) else
        {
            return
        }

        let request = VNCoreMLRequest(model: model)
        {
            (finishedRequest, err) in

            // check the error

            guard let results = finishedRequest.results as? [VNClassificationObservation] else
            {
                return
            }
            
            guard let firstObservation = results.first else
            {
                return
            }
            
            print(firstObservation.identifier, firstObservation.confidence)
            DispatchQueue.main.async {
                let confidencePercent = String(format: "%.2f", firstObservation.confidence * 100)
                self.classificationLabel.text = firstObservation.identifier + ": " + confidencePercent + "%"
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
}


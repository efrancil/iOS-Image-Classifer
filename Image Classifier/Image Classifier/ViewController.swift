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
        print("Camera was able to capture a frame:", Date())
//        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(<#T##sbuf: CMSampleBuffer##CMSampleBuffer#>) else
//        {
//            return
//        }
//
//        guard let model_CIFAR10 = try? VNCoreMLModel(for: CIFAR_10_First_CNN(configuration: .init()).model) else
//        {
//            return
//        }
//
//        let request = VNCoreMLRequest(model: model_CIFAR10)
//        {
//            (finishedRequest, err) in
//
//            // check the error
//
//            print(finishedRequest.results)
//        }
//
//        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [ :]).perform([request])
    }
}


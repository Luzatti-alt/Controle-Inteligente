package com.lucas.controlei

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: ImageView
    private lateinit var tvDeteccoes: TextView
    private lateinit var detector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService

    // Controle para não processar frames em paralelo
    @Volatile private var processando = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        tvDeteccoes = findViewById(R.id.tvDeteccoes)

        detector = YoloDetector(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (temPermissao()) {
            iniciarCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    private fun temPermissao() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            iniciarCamera()
        }
    }

    private fun iniciarCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview normal da câmera
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // ImageAnalysis para capturar frames e rodar YOLO
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetRotation(previewView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                if (!processando) {
                    processando = true
                    processarFrame(imageProxy)
                } else {
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e("Camera", "Erro ao iniciar câmera", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processarFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()

            // Rodar detecção (já está em thread separada via cameraExecutor)
            val (anotado, deteccoes) = detector.detectar(bitmap)

            // Atualizar UI na thread principal
            runOnUiThread {
                overlayView.setImageBitmap(anotado)
                tvDeteccoes.text = if (deteccoes.isEmpty()) {
                    "Nenhuma detecção"
                } else {
                    deteccoes.joinToString("\n")
                }
            }

        } catch (e: Exception) {
            Log.e("YOLO", "Erro ao processar frame", e)
        } finally {
            imageProxy.close()
            processando = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.fechar()
        cameraExecutor.shutdown()
    }
}
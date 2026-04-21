package com.lucas.controlei

import ai.onnxruntime.*
import android.content.Context
import android.graphics.*
import java.nio.FloatBuffer

class YoloDetector(context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName = "images"

    val classNames = listOf("Desodorante Azul", "Garrafa")
    val classColors = listOf(
        Color.rgb(0, 120, 255),   // Azul para Desodorante Azul
        Color.rgb(0, 200, 80)     // Verde para Garrafa
    )

    companion object {
        const val INPUT_SIZE = 640
        const val CONF_THRESHOLD = 0.1f
        const val IOU_THRESHOLD = 0.45f
        const val MODEL_FILE = "modelo_opset21.onnx"
    }

    data class Detection(
        val classId: Int,
        val className: String,
        val confidence: Float,
        val box: RectF  // coordenadas em pixels do frame original
    )

    init {
        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        val opts = OrtSession.SessionOptions()
        session = ortEnv.createSession(modelBytes, opts)
    }

    fun detectar(bitmap: Bitmap): Pair<Bitmap, List<String>> {
        val originalW = bitmap.width.toFloat()
        val originalH = bitmap.height.toFloat()

        // 1. Pré-processar: redimensionar para 640x640 e normalizar
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = bitmapToFloatBuffer(resized)

        // 2. Rodar inferência
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            inputBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        val results = session.run(mapOf(inputName to inputTensor))
        val rawOutput = (results[0].value as Array<Array<FloatArray>>)[0] // [6][8400]

        // 3. Parsear saída (formato YOLO: [cx,cy,w,h, conf_cls0, conf_cls1] x 8400)
        val deteccoes = parsearSaida(rawOutput, originalW, originalH)

        // 4. Aplicar NMS
        val finais = nms(deteccoes)

        // 5. Desenhar boxes no bitmap original
        val anotado = desenharBoxes(bitmap, finais)

        // 6. Formatar lista de detecções
        val lista = finais.map { "${it.className}: ${"%.2f".format(it.confidence * 100)}%" }

        inputTensor.close()
        results.close()

        return Pair(anotado, lista)
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // YOLO espera: [1, 3, 640, 640] em ordem RGB normalizado 0..1
        // Ordem: todos os R, depois todos os G, depois todos os B
        val buffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rArr = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val gArr = FloatArray(INPUT_SIZE * INPUT_SIZE)
        val bArr = FloatArray(INPUT_SIZE * INPUT_SIZE)

        for (i in pixels.indices) {
            val px = pixels[i]
            rArr[i] = (px shr 16 and 0xFF) / 255f
            gArr[i] = (px shr 8 and 0xFF) / 255f
            bArr[i] = (px and 0xFF) / 255f
        }

        buffer.put(rArr)
        buffer.put(gArr)
        buffer.put(bArr)
        buffer.rewind()
        return buffer
    }

    private fun parsearSaida(
        output: Array<FloatArray>, // [6][8400]
        originalW: Float,
        originalH: Float
    ): List<Detection> {
        val deteccoes = mutableListOf<Detection>()
        val numDetections = output[0].size // 8400

        val scaleX = originalW / INPUT_SIZE
        val scaleY = originalH / INPUT_SIZE

        for (i in 0 until numDetections) {
            val cx = output[0][i]
            val cy = output[1][i]
            val w  = output[2][i]
            val h  = output[3][i]

            // Encontrar classe com maior confiança
            var maxConf = 0f
            var maxCls = 0
            for (c in classNames.indices) {
                val conf = output[4 + c][i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxCls = c
                }
            }

            if (maxConf < CONF_THRESHOLD) continue

            // Converter de coordenadas do input 640x640 para coordenadas originais
            val x1 = (cx - w / 2) * scaleX
            val y1 = (cy - h / 2) * scaleY
            val x2 = (cx + w / 2) * scaleX
            val y2 = (cy + h / 2) * scaleY

            deteccoes.add(
                Detection(
                    classId = maxCls,
                    className = classNames[maxCls],
                    confidence = maxConf,
                    box = RectF(
                        x1.coerceAtLeast(0f),
                        y1.coerceAtLeast(0f),
                        x2.coerceAtMost(originalW),
                        y2.coerceAtMost(originalH)
                    )
                )
            )
        }

        return deteccoes
    }

    private fun nms(deteccoes: List<Detection>): List<Detection> {
        val sorted = deteccoes.sortedByDescending { it.confidence }.toMutableList()
        val resultado = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            resultado.add(best)
            sorted.removeAll { iou(best.box, it.box) > IOU_THRESHOLD && it.classId == best.classId }
        }

        return resultado
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = maxOf(a.left, b.left)
        val interTop    = maxOf(a.top, b.top)
        val interRight  = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - interArea

        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    private fun desenharBoxes(bitmap: Bitmap, deteccoes: List<Detection>): Bitmap {
        val resultado = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultado)

        val paintBox = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val paintBg = Paint().apply {
            style = Paint.Style.FILL
        }
        val paintText = Paint().apply {
            color = Color.WHITE
            textSize = 36f
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
        }

        for (det in deteccoes) {
            val cor = classColors[det.classId]
            paintBox.color = cor
            canvas.drawRect(det.box, paintBox)

            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val textW = paintText.measureText(label)
            val textH = paintText.textSize

            // Fundo do label
            paintBg.color = cor
            val labelTop = maxOf(det.box.top - textH - 8f, 0f)
            canvas.drawRect(
                det.box.left,
                labelTop,
                det.box.left + textW + 12f,
                labelTop + textH + 8f,
                paintBg
            )

            canvas.drawText(label, det.box.left + 6f, labelTop + textH, paintText)
        }

        return resultado
    }

    fun fechar() {
        session.close()
        ortEnv.close()
    }
}
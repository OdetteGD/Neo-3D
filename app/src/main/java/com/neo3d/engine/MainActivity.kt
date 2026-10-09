package com.neo3d.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity(), SurfaceHolder.Callback {
    private external fun nativeStart(surface: Surface)
    private external fun nativeResize(width: Int, height: Int)
    private external fun nativeStop()
    private external fun nativeStatus(): String
    private external fun nativeLoadGlb(data: ByteArray): String
    private external fun nativeOrbit(dx: Float, dy: Float)
    private external fun nativeSetAutoRotate(enabled: Boolean)
    private external fun nativeResetView()
    private lateinit var status: TextView
    private var surfaceReady = false
    private var lastX = 0f
    private var lastY = 0f
    private var autoRotate = true
    private val importRequestCode = 3107

    companion object { init { System.loadLibrary("neo3d") } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_LAYOUT_STABLE)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xFF10151D.toInt()) }
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(10, 4, 10, 4) }
        status = TextView(this).apply { text = "NEO-3D | Vulkan 3D editor"; textSize = 12f; setTextColor(0xFFE3EAF4.toInt()); layoutParams = LinearLayout.LayoutParams(0, -2, 1f) }
        val rotate = Button(this).apply {
            text = "Auto: ON"
            setOnClickListener { autoRotate = !autoRotate; nativeSetAutoRotate(autoRotate); text = if (autoRotate) "Auto: ON" else "Auto: OFF" }
        }
        val reset = Button(this).apply { text = "Reset view"; setOnClickListener { nativeResetView(); autoRotate = true; rotate.text = "Auto: ON"; status.text = nativeStatus() } }
        val refresh = Button(this).apply { text = "GPU"; setOnClickListener { status.text = nativeStatus() } }
        val import = Button(this).apply {
            text = "Import GLB"
            setOnClickListener {
                val picker = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("model/gltf-binary", "application/octet-stream"))
                }
                startActivityForResult(picker, importRequestCode)
            }
        }
        bar.addView(status); bar.addView(import); bar.addView(rotate); bar.addView(reset); bar.addView(refresh)
        val viewport = SurfaceView(this)
        viewport.setBackgroundColor(0xFF10151D.toInt())
        viewport.holder.addCallback(this)
        viewport.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> { lastX = event.x; lastY = event.y; true }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.x - lastX
                    val dy = event.y - lastY
                    lastX = event.x; lastY = event.y
                    if (surfaceReady) { nativeSetAutoRotate(false); autoRotate = false; rotate.text = "Auto: OFF"; nativeOrbit(dx, dy) }
                    true
                }
                else -> true
            }
        }
        root.addView(bar, LinearLayout.LayoutParams(-1, -2))
        root.addView(viewport, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    @Deprecated("Uses the platform document picker result callback for broad Android compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != importRequestCode || resultCode != RESULT_OK || data?.data == null) return
        try {
            val bytes = contentResolver.openInputStream(data.data!!)?.use { input ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(64 * 1024)
                var total = 0
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    total += count
                    if (total > 128 * 1024 * 1024) throw IllegalArgumentException("GLB exceeds 128 MiB limit")
                    output.write(buffer, 0, count)
                }
                output.toByteArray()
            } ?: throw IllegalStateException("Unable to open selected file")
            if (!surfaceReady) throw IllegalStateException("Wait for the 3D viewport to start before importing")
            status.text = nativeLoadGlb(bytes)
        } catch (e: Exception) {
            status.text = "Import failed: ${e.message ?: "unable to read file"}"
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) { nativeStart(holder.surface); surfaceReady = true; status.text = nativeStatus() }
    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) { if (surfaceReady) nativeResize(width, height) }
    override fun surfaceDestroyed(holder: SurfaceHolder) { surfaceReady = false; nativeStop() }
    override fun onDestroy() { if (surfaceReady) nativeStop(); surfaceReady = false; super.onDestroy() }
}

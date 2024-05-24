package com.surendramaran.birdDetectionApp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class InformationActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_information)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val backButton = findViewById<Button>(R.id.backButton)
        backButton.setOnClickListener{showMainActivity()};

        val data = intent.getStringArrayExtra("birdInfo")
        val splitData = data?.joinToString("| ")
        val convertedArray = splitData?.split("| ")?.toTypedArray()

        if (convertedArray != null){
            val speciesNameTextView = findViewById<TextView>(R.id.birdName)
            val speciesHabitatTextView = findViewById<TextView>(R.id.birdHabitat)
            val speciesInfoTextView = findViewById<TextView>(R.id.birdInfo)

            speciesNameTextView.text = convertedArray[0]
            speciesHabitatTextView.text = "They can be found " + convertedArray[2]
            speciesInfoTextView.text = convertedArray[1]
        }
    }

    private fun showMainActivity() {
        val intent = Intent(this, MainActivity::class.java)

        startActivities(arrayOf(intent))
    }
}
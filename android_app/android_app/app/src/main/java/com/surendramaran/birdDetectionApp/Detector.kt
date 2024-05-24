package com.surendramaran.birdDetectionApp

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private var speciesName : String = ""
    private var speciesInfo : String = ""
    private var speciesHabitat : String = ""

    var overrideableBirdInfo: Array<String> = arrayOf();

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun setBirdInformation(bird: String) {
        return when (bird) {
            "African spoonbill" -> {
                speciesName = "African Spoonbill (Platalea alba)"
                speciesInfo = "African Spoonbills live in marshy wetlands with open shallow waters. " +
                        "They feed inside shallow waters on various fish, molluscs, insects, crustaceans and amphibians. " +
                        "They do this by swinging their open bill in the water from side-to-side. " +
                        "African Spoonbills' breeding season lasts from the Winter to the Spring and the eggs are laid typically " +
                        "during April or May. The eggs are incubated for up to 29 days and the new born are cared for by both parents " +
                        "for around a month. Afterwards, the birds can leave the nest and begin fledging after another four weeks."
                speciesHabitat = "mostly in Africa and Madagascar"
            }
            "Budgerigar" -> {
                speciesName = "Budgerigar (Melopsittacus undulatus)"
                speciesInfo = "Budgerigars are small, long-tailed, seed-eating parrots which are naturally green and yellow with black. " +
                        "However, they have also been bred to come in varieties of blues, whites, yellows, greys and even with small crests. " +
                        "They are the third most common pet behind the dog and cat due to their small size, low cost and ability to mimic human speech. " +
                        "Budgerigars breed when the opportunity presents itself and in pairs; Laying their eggs in nests set in holes of trees, fences and logs. " +
                        "The eggs are laid on alternating days with a two day gap between each egg, laying a total of around 6 to 8 eggs. The eggs are incubated " +
                        "for 18 to 21 days and fledging about a month after hatching."
                speciesHabitat = "in dried parts of Australia"
            }
            "Carrion Crow" -> {
                speciesName = "Carrion Crow (Corvus corone)"
                speciesInfo = "Closely related to the hooded crow (Corvus cornix), the carrion crow is larger and of a stockier build. " +
                        "This species is largely solitary and sociable but occasionally nests in isolated trees. Like all other corvids, " +
                        "Carrion crows are extremely intelligent; Enough that they can identify different numbers up to 30 and recognise human " +
                        "and crow faces. Carrion crows feed on insects, earthworms, grain, fruit, seeds, nuts, small mammals, amphibians, fish and even " +
                        "may steal eggs. They also harass birds of prey and foxes for their kills. Carrion crows have little natural predators but some " +
                        "larger birds such as the Eurasian eagle-owl hunt them. Carrion crows usually nest in tall trees but have also used cliff ledges and old buildings. " +
                        "They lay 3 to 4 brown dotted blue or greenish eggs that incubate for 18 to 20 days by the female alone who's fed by the male. Afterwards, " +
                        "the young fledge after around a month."
                speciesHabitat = "throughout Eurasia"
            }
            "Cattle Egret" -> {
                speciesName = "Cattle egret (Bubulgus)"
                speciesInfo = "Cattle egrets are sometimes split between two species, being the western cattle egret and eastern cattle egret. " +
                        "They usually have white plumage but develop orange plumes on their backs, breasts and crowns during the breeding season. " +
                        "Additionally, their bills, legs and irises become bright red for a short time before pairing with a mate. This species " +
                        "has went through one of the largest and fastest natural expansions of any bird species originally being only native " +
                        "to sections of southern Spain and Portugal. Cattle egrets nest in colonies within woodlands, generally near bodies of water. " +
                        "The breeding season varies depending on location; For example, Cattle egret in northern india experience their breeding season in " +
                        "May whilst Australian Cattle egret experience it in November to Early January. Pairs form within 3 to 4 days and a new mate is chosen " +
                        "each season or after experiencing nest failure. Their incubation period lasts around 23 days with both sexes sharing incubation duties. " +
                        "Eventually, the chicks become independent around a month and a half after hatching. Cattle egret can feed on bugs, fish, worms, lizards and " +
                        "snakes. They can be found accompanying cattle and other large mammals to catch the insects that are disturbed by their movements."
                speciesHabitat = "throughout Tropics, Subtropics and Warm-temperature zones"
            }
            "Cockatiel" -> {
                speciesName = "Cockatiel (Nymphicus hollandicus)"
                speciesInfo = "Cockatiels are the only member of the genus Nymphicus. Cockatiels currently have 22 colour mutations, 8 " +
                        "of which are only in Australia. However, the most common colour scheme is a yellow head with a grey body. Before they moult for the first time, " +
                        "male and female cockatiels are almost indistinguishable. After moulting, males lose the white lining and spots on the underside of their tail " +
                        "feathers and wings. Additionally, the grey feathers on their cheeks and crest are replaced by yellow feathers whilst the orange cheeks become brighter. " +
                        "Cockatiels breed during seasonal rainfall and nest in tree hollows near fresh water sources, usually picking eucalyptus/gum trees. 4 to 7 eggs " +
                        "are laid which are incubated by the female for 17 to 23 days. Afterwards, the chicks fledge after 5 weeks. Wild cockatiels usually eat seeds " +
                        "and cultivated crops and live for around 12 to 15 years. Although in captivity, this can go up to 16 to 25 years within good living conditions."
                speciesHabitat = "throughout Australian wetlands, scrublands and bushlands"
            }
            "Duck" -> {
                speciesName = "American Pekin (Anas platyrhynchos domensticus)"
                speciesInfo = "15 birds from a breed of mallard in China roughly translated as the 'Ten-Pound Duck' were shipped from china " +
                        "to the United States in 1874, of which only 4 survived. The 3 hens then laid more than 300 eggs, becoming the foundation " +
                        "for the American Pekin breed. These ducks are raised mostly for their meat and they represent more than half of all ducks " +
                        "that are raised for slaughter in America. American Pekins can lay over 150 eggs per year. Although, the eggs may have to be " +
                        "artificially incubated as they are not good at incubating their eggs. As they were bred as a food source, they have a high " +
                        "feed conversion ratio, meaning that they are easily convert feed into body mass."
                speciesHabitat = "bred and populated in America"
            }
            "Eurasian Eagle-owl" -> {
                speciesName = "Eurasian eagle-owl (Bubo bubo)"
                speciesInfo = "The Eurasian Eagle-owl is one of the largest species of owls with their females growing up to a length of 75 cm. " +
                        "These Eagle-owls are nocturnal predators, mostly hunting small mammals and other birds. They may also hunt reptiles, amphibians, " +
                        "large insects, invertebrates and fish on occasion. Eurasian Eagle-owls are also some of the most widely distributed species of owl, " +
                        "having a range of around 51.4 million km² across Europe and Asia with a population of between 100,000 and 500,000. Although distributed " +
                        "sparsely, they can inhabit a large range of habitats; Ranging from deserts' edges to coniferous forests, excluding extreme environments such as " +
                        "humid rainforests and arctic tundras. Eurasian eagle-owls' mating system occurs in January or February and even though they pair with a mate for " +
                        "life, they will still participate in courtship rituals; This is done likely to re-affirm the bond with their pair. Nests are then advertised by the male " +
                        "to the female by kneading a small hole where possible and making clucking noises. The female will then choose the nesting location. Eagle owl's eggs are laid " +
                        "at intervals of 3 days and incubate from mid-January to mid-March. Afterwards, the hatchlings fledge from mid-April to August."
                speciesHabitat = "throughout Eurasia"
            }
            "Flamingo" -> {
                speciesName = "Flamingo (Phoenicopterus)"
                speciesInfo = "There are 6 different species of the flamingo. Flamingos generally stand on one leg with the other hidden under their body. " +
                        "However, the reason for this is unknown. Additionally, flamingos legs appear to bend backwards when they move as the middle joint on their leg " +
                        "is their ankle instead of their knee. Flamingos are born with gray-ish red feathers which changes to a hue between a light pink to a bright red due " +
                        "to aqueous bacteria and beta-carotene consumed from their diet. Flamingos filter-feed (they strain the food out of the water) on brine shrimp and " +
                        "blue-green algae as well as other small creatures in the water, making them omnivorous. They filter the food using a hair-like material lined around " +
                        "mandibles and tongue called lamellae. Flamingos live in colonies which can number up to the thousands. Before mating, these colonies shrink into groups of " +
                        "around 15 to 50 birds. Afterwards, the birds display themselves to each other to find a mate. Once they find a mate, they bond and nest. " +
                        "For the first 6 days after the chick hatches, the adults stay in the nesting sites. At the 7 to 12 day point, the chicks begin to explore their " +
                        "surroundings outside the nest. After a week, chicks gather together in groups called microcrèches to avoid predators. Lastly, after 3 to 3.5 months, " +
                        "they grow their flight feathers."
                speciesHabitat = "as 4 species throughout the americas and 2 species native to Afro-Eurasia"
            }
            "Little Egret" -> {
                speciesName = "Little egret (Egretta garzetta)"
                speciesInfo = "Little egret are aquatic birds which feed in shallow waters. The species was turned locally extinct in western europe during the 19th century due to their feathers " +
                        "being used as hat decorations. However, they have begun to reemerge in France, the Netherlands, Ireland and Britain as they had been found breeding in those locations by " +
                        "the start of the 21st century. They have also been noted to be spreading to the west as they have been found in Barbados in 1954. The northern populations of Little egrets are " +
                        "migratory and travel to africa or sometimes staying in southern europe. Meanwhile, some Asian populations migrate towards the Philippines. " +
                        "For feeding, Little egrets sometimes stalk their prey in shallow water. This is done by either forcing a disturbance by running with raised wings or standing still " +
                        "and waiting to ambush their prey. Their diet consists mainly of fish but they may also eat amphibians, small reptiles, mammals, birds, worms and crustaceans. " +
                        "They nest on colonies and lay 3 to 5 eggs. The eggs are incubated by both parents for 21 to 25 days before hatching. Afterwards, the hatchling is cared for until they can fly " +
                        "after around 40 to 45 days."
                speciesHabitat = "throughout southern Europe, the Middle East, Africa and southern Asia"
            }
            "Love Bird" -> {
                speciesName = "Lovebird (Agapornis)"
                speciesInfo = "Lovebirds can form relationships with both people and other lovebirds that can last a lifetime. However, " +
                        "unless a gentle bond is established they may show aggression and bite. If someone wants a lovebird, it's recommended to " +
                        "obtain one through breeding as wild lovebirds may contain diseases and also mourn the loss of their flock. For feeding, captive " +
                        "lovebirds can make due with a mix of different kinds of seeds, grains and nuts and optionally fruit and/or vegetables. When the lovebird " +
                        "makes its nest, it will mate and lay eggs after 3 to 5 days. The female lovebird will stay in the nesting box for hours before laying the eggs. " +
                        "After the first egg, following eggs will be laid every other day. Usually only around 4 to 6 eggs are laid. Sometimes lovebirds may even lay eggs without either a " +
                        "nest or a mate."
                speciesHabitat = "native to Africa with the grey-headed variant native to Madagascar"
            }
            "Northern bald ibis" -> {
                speciesName = "Northern bald ibis (Geronticus eremita)"
                speciesInfo = "The Northern bald ibis used to be commonly found all over the Middle East, northern Africa, Southern and central europe. Unfortunately, " +
                        "the species disappeared from Europe over 300 years ago. However, there are programs that are reintroducing them. These birds are usually silent except " +
                        "for when they are participating in a breeding colony. In these breeding colonies, the males with longer bills are more successful at attracting mates. " +
                        "The Northern bald ibiis set up their nests on cliff edges or around boulders in steep areas. They begin to breed when they are in an age range of " +
                        "3 - 5 years old and pair for life. After breeding, 2 - 4 eggs blue-white with brown spotted eggs are laid which turn brown during incubation. The eggs " +
                        "are incubated for 24-25 days and the hatchlings fledge after another 40 to 50 days. To feed, this birds commute from their breeding sites to the feeding location " +
                        "in a V formation. They consume many kinds of animals such as lizards, beetles, ground nesting birds, invertebrates and small mammals. They can scavenge up to 15km away " +
                        "from their colony."
                speciesHabitat = "throughout Morocco and Turkey"
            }
            "Peafowl" -> {
                speciesName = "Peafowl (Pavo)"
                speciesInfo = "Peafowls' large 'tail' or 'train' has had many scientific debates regarding its purpose. Charles Darwin thought " +
                        "that their purpose was to attract females; Whilst more recently, Amotz Zahavi proposed that they function as signals to " +
                        "indicate the peafowl's health as a weaker peafowl would struggle to survive with the large structure. Female peafowls (peahens) " +
                        "have smaller plumage than male peafowls (peacocks). However, it's been noted that mature peahens have been seen suddenly growing " +
                        "male peafowl trains and making male calls. It has been suggested that this is due to a lack of estrogen being produced from old and/or " +
                        "damaged ovaries. This also suggests that the male train and calls are the default without hormonal intervention. Peafowls are omnivores and " +
                        "mostly eat flower petals, seeds heads, plants, arthropods, insects, reptiles and amphibians. Domesticated peafowls can also eat bread and " +
                        "cracked grains like corn, oats, cheese and cooked rice. It's been also noted that they enjoy protein-rich foods such as larvae and different kinds of " +
                        "meat and fruit. Peahens usually lay eggs around mid April after they reach maturity after around 2 years. However, they've been seen laying " +
                        "eggs after only a year nearing the end of their first summer."
                speciesHabitat = "across India and dried lowland areas of Sri Lanka"
            }
            "Sacred ibis" -> {
                speciesName = "African sacred ibis (Threskiornis aethiopicus)"
                speciesInfo = "African sacred ibis are found in marshy wetlands and mudflats. They prefer to nest on trees that are in " +
                        "or near bodies of water. They feed in shallow wetlands and wet pastures with soft soil on mainly insects, " +
                        "worms, crustaceans, molluscs, fish, reptiles and most invertebrates they can find. They have even been seen " +
                        "looking for food in cultivation and rubbish dumps. They generally feed during the day in flocks. African sacred ibis " +
                        "breed once per year in the wet season. This means that it's from March to August when they're in Africa or April to May " +
                        "when they're in Iraq. The birds nest in a stick nest in trees and partake in tree colonies, often with other large birds " +
                        "like herons, African spoonbills and storks. The females lay 1 to 5 eggs per season which are incubated by both parents for " +
                        "21 to 29 days. Once the eggs hatch, one parent will take care of the hatchlings for the first week. The hatchlings then fledge " +
                        "after 35 to 40 days and are independent after 44 to 48 days."
                speciesHabitat = "throughout most of Africa as well as small parts of Iraq, Iran and Kuwait"
            }
            "Snow owl" -> {
                speciesName = "Snowy owl (Bubo scandiacus)"
                speciesInfo = "Snowy owls are a specialized nocturnal hunter species that prey on tundra-dwelling lemmings but can hunter most other prey during " +
                        "non breeding seasons. This means that the owl's population scales with the lemmings' availability. Snowy owls are nomadic, meaning that " +
                        "they solemn breed twice in the same place or with the same mate. If there are no prey, they may not even breed at all. They nest on small " +
                        "raised areas on tundra grounds and typically lay 5 to 11 eggs during early May to the first 10 days of June. A full clutch could take up to " +
                        "a month to lay with time between every egg being staggered. The hatchlings usually become independent in Autumn. The migration patterns of " +
                        "the Snowy owls are unpredictable as they can traverse almost anywhere that is close to the Arctic. They have even been seen migrating south " +
                        "in large numbers. Due to the difficult to interpret patterns, there is a low amount of historical information regarding the species' status. " +
                        "However, recent data reads that the population is steadily declining, estimating to have halved over a long period of time."
                speciesHabitat = "throughout the Arctic regions of both North America and the Palearctic."
            }
            "Straw necked ibis" -> {
                speciesName = "Straw-necked ibis (Threskiornis spinicollis)"
                speciesInfo = "Straw-necked ibises are only partially a migratory species. Some of them are sedentary whilst others make seasonal or random movements when " +
                        "the conditions of the water are different. They are nomadic and are moving constantly whilst attempting to find suitable habitats. They are usually " +
                        "seen atop high branches of bare trees. Straw-necked ibises typically feed on invertebrates but their diets may vary. In shallow water, they eat molluscs, " +
                        "frogs, aquatic insects, frogs, freshwater crayfish and fish; Whilst on land, they eat land-based insects. Additionally, they have even been seen flicking " +
                        "toxic toads around until they release their defensive toxin so they could eat them afterwards. Their breeding season depends mainly on water conditions. " +
                        "In southwestern Australia, it normally happens during the August to December months. Meanwhile, breeding in the north happens on a much lower scale. " +
                        "Straw-necked ibises build a large cup-shaped stick nest and plants. Afterwards, they breed in colonies, often with another ibis, the Australian white ibis. " +
                        "The nests are used yearly and they lay around 2 to 5 eggs. The eggs incubate for around 24 to 25 days by both parents. Lastly, the child is taken care of by " +
                        "both parents for about 35 days after hatching with feeding via regurgitation and can last for up to 2 weeks after leaving the nest."
                speciesHabitat = "throughout Australia, New Guinea and parts of Indonesia"
            }
            else -> throw AssertionError()
        }
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime


        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        overrideableBirdInfo = arrayOf<String>(speciesName, speciesInfo, speciesHabitat)
        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                var clsName = labels[maxIdx] // Returns the name of the label
                if (speciesName == ""){
                    speciesName = clsName
                }
                if (clsName == "Sacred Iblis"){
                    clsName = "Sacred ibis"
                }
                else if (clsName == "Straw necked Iblis"){
                    clsName = "Straw necked ibis"
                }
                else if (clsName == "Northern bald iblis"){
                    clsName = "Northern bald ibis"
                }
                else if (clsName == "Peacock"){
                    clsName = "Peafowl"
                }

                setBirdInformation(clsName)
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}
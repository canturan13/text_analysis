import requests
import time

test_data =[
"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
"Etiam at leo vestibulum enim tincidunt dapibus.",
"Etiam quis nulla vulputate, malesuada nisi nec, placerat diam.",
"Suspendisse hendrerit ante vitae dolor tristique, at hendrerit dui placerat.",
"Ut tempus velit non elementum convallis.",
"Donec eu tortor ut lorem tempor lobortis.",
"Nunc ornare mauris vel nulla tempor porttitor.",
"In suscipit ex eu ultricies aliquam.",
"In nec nibh euismod, efficitur purus et, faucibus velit.",
"Pellentesque sagittis ipsum vel tortor luctus tincidunt.",
"Vivamus venenatis dui bibendum diam vulputate, ac placerat orci malesuada.",
"Fusce tincidunt magna ac lorem egestas tincidunt.",
"Pellentesque vitae velit ut urna accumsan scelerisque.",
"Phasellus sollicitudin dui a turpis faucibus, nec congue velit feugiat.",
"Donec ullamcorper arcu et rutrum faucibus.",
"Sed hendrerit mi nec urna lobortis, in lacinia neque ultrices.",
"Curabitur in urna nec mauris consequat iaculis ac venenatis eros.",
"Sed non purus et nibh tincidunt rutrum at nec eros.",
"Praesent semper urna quis lacus gravida, a eleifend lorem dapibus.",
"Cras fermentum leo vel congue consectetur.",
"Aliquam pellentesque quam id mauris semper tristique.",
"Mauris non lorem non turpis feugiat consequat ut ac tellus.",
"Proin non tortor egestas, luctus nisi vitae, ornare ex.",
"Sed tincidunt risus ut aliquam imperdiet.",
"Integer sit amet ligula vitae dui aliquam eleifend.",
"Phasellus semper turpis ac urna feugiat, nec rhoncus nibh aliquet.",
"Nulla pulvinar enim ut neque tincidunt, nec eleifend magna congue.",
"Nullam nec magna nec sem bibendum elementum nec eu nulla.",
"Pellentesque ac dui sed eros ullamcorper tempus.",
"Sed eu nibh suscipit, volutpat lorem sed, bibendum ex.",
"Mauris vel est at urna convallis placerat vitae et neque.",
"Ut id purus a nibh lacinia ullamcorper.",
"Sed posuere arcu ultricies massa malesuada, sit amet hendrerit tortor condimentum.",
"Praesent blandit lacus id mi auctor gravida.",
"Suspendisse volutpat urna vel dui laoreet gravida.",
"In ut lectus in nisi maximus interdum.",
"In sed nulla gravida, suscipit magna placerat, feugiat nibh.",
"Sed ultrices ante eu nibh suscipit, id facilisis arcu sollicitudin.",
"Nunc tristique magna in euismod dignissim.",
"Proin vestibulum nunc vel ex fringilla, vel imperdiet purus euismod.",
"Phasellus posuere felis eget lacus finibus vehicula.",
"Etiam euismod sapien sed mi auctor, ut consectetur risus fringilla.",
"Praesent viverra nunc in nunc luctus facilisis.",
"Donec ultricies elit ut ex posuere condimentum.",
"Integer non purus et sem tempus accumsan a vitae ipsum.",
"Phasellus at enim quis nisl gravida faucibus et vitae nulla.",
"Mauris dapibus est vitae quam porttitor volutpat et nec lorem.",
"Quisque tincidunt nisl fermentum iaculis malesuada.",
"Etiam sodales metus et est efficitur blandit.",
"Fusce a turpis id tellus rhoncus venenatis.",
"Aenean id purus ut tortor pretium molestie.",
"Maecenas at enim scelerisque, dapibus lorem quis, lobortis dui.",
"Ut pulvinar dolor id sagittis molestie.",
"Vivamus ac metus mattis, vestibulum magna eu, consectetur massa.",
"Donec viverra turpis congue, commodo diam sit amet, egestas lectus.",
"Phasellus consequat urna a lectus eleifend elementum.",
"Nullam at libero tempus, iaculis augue efficitur, euismod metus.",
"Ut rutrum nibh ac massa vehicula facilisis.",
"Fusce vel tellus eget justo faucibus luctus non tempor erat.",
"Sed bibendum nulla nec enim vehicula, ac tempus nibh accumsan.",
"Ut lobortis metus sed mauris molestie maximus.",
"Pellentesque pellentesque arcu vitae urna ullamcorper, eget congue leo consequat.",
"Pellentesque euismod mauris vel ante pretium, ut gravida nisl rhoncus.",
"Cras vulputate velit at nunc sollicitudin, et tincidunt ante malesuada.",
"In vehicula urna eu tortor lobortis facilisis.",
"Aenean fringilla massa eget elementum mollis.",
"Vestibulum accumsan enim et sem viverra, a fringilla magna fermentum.",
"Suspendisse ut arcu auctor, suscipit dui a, congue quam.",
"Suspendisse eu arcu lobortis, blandit nisi commodo, ullamcorper ex.",
"Vestibulum eget quam imperdiet, feugiat odio at, ultricies orci.",
"Curabitur pellentesque sem in ultrices semper.",
"Nullam malesuada erat quis metus euismod dapibus.",
"Morbi vitae tellus scelerisque, vulputate lorem et, scelerisque elit.",
"Curabitur cursus velit nec justo rutrum, ut pulvinar elit scelerisque.",
"Ut commodo quam non mi pulvinar ultricies.",
"Integer sit amet odio sit amet lacus hendrerit viverra.",
"In scelerisque quam in metus consequat, in ultrices neque scelerisque.",
"Morbi faucibus elit in turpis rhoncus dapibus.",
"Pellentesque vel dolor a purus porttitor pulvinar eget sed enim."
]


local_jina_times = []
loop=1
url = "http://127.0.0.1:8000/v1/embeddings"

local_jina_times = []
loop = 1
for i in range(loop):
    for sentence in test_data:
        try:
            start = time.time()
            response = requests.post(
                url,
                json=[sentence],
                headers={"accept": "application/json", "Content-Type": "application/json"},
            )
            end = time.time()
            local_jina_times.append(end - start)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
print("Local JINA (CUDA) Ortalama Yanıt Süresi:", sum(local_jina_times) / len(local_jina_times))

import numpy as np
import matplotlib.pyplot as plt
print(local_jina_times)

stats = {
    "Local JINA MAC": {
        "mean": np.mean(local_jina_times) if local_jina_times else 0,
        "median": np.median(local_jina_times) if local_jina_times else 0,
        "std": np.std(local_jina_times) if local_jina_times else 0,
        "min": np.min(local_jina_times) if local_jina_times else 0,
        "max": np.max(local_jina_times) if local_jina_times else 0,
    },
}

methods = list(stats.keys())
means = [stats[method]["mean"] for method in methods]
medians = [stats[method]["median"] for method in methods]
stds = [stats[method]["std"] for method in methods]
mins = [stats[method]["min"] for method in methods]
maxs = [stats[method]["max"] for method in methods]

plt.figure(figsize=(12, 8))
plt.plot(methods, means, marker='o', linestyle='-', label='Mean', color='blue')
plt.plot(methods, medians, marker='o', linestyle='--', label='Median', color='green')
plt.plot(methods, stds, marker='o', linestyle='-.', label='Std Dev', color='orange')
plt.plot(methods, mins, marker='o', linestyle=':', label='Min', color='red')
plt.plot(methods, maxs, marker='o', linestyle='-', label='Max', color='purple')

plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
plt.savefig('statistical_analysis_plot.png')

for method, values in stats.items():
    print(f"{method}:")
    for stat, value in values.items():
        print(f"  {stat.capitalize()}: {value:.4f}")


plt.figure(figsize=(8, 6))
plt.plot(local_jina_times, marker='o', linestyle='-', color='purple', label='Local JINA Response Times')
plt.title('Local JINA MAC Response Times')
plt.xlabel('Number of Requests')
plt.ylabel('Response Time (seconds)')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
plt.savefig('local_jina_response_times.png')
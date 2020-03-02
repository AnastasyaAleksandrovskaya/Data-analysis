import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import time
from streamlit import config
import random
import plotly.graph_objs as go
from io import BytesIO
import requests
from random import random
import os
import platform
import sys
from collections import namedtuple


# pip install plotly

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)
"""
# Uber Pickups in New York City
"""
"""
"""

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    return data


data = load_data(100000)
'data',data
demo = st.sidebar.selectbox("Choose demo", ["Code", "Animation", "Caching", "Plotly example", "Reference",
                                            "Run on save", "Syntax error", "Syntax hilite", "Video"

                                            ], 1)
hour=st.sidebar.slider('hour',0,23,10)
# hour=st.sidebar.number_input('hour',0,23,10)
data=data[data[DATE_TIME].dt.hour==hour]

'## Geo Data at %sh' % hour
st.map(data)
''
midpoint = (np.average(data["lat"]), np.average(data["lon"]))

if demo == "Caching":
    cache_was_hit = True


    @st.cache
    def check_if_cached():
        global cache_was_hit
        cache_was_hit = False


    @st.cache
    def my_func(arg1, arg2=None, *args, **kwargs):
        return random.randint(0, 2 ** 32)


    check_if_cached()

    if cache_was_hit:
        st.warning("You must clear your cache before you run this script!")
        st.write(
            """
            To clear the cache, press `C` then `Enter`. Then press `R` on this page
            to rerun.
        """
        )
    else:
        st.warning(
            """
            IMPORTANT: You should test rerunning this script (to get a failing
            test), then clearing the cache with the `C` shortcut and checking that
            the test passes again.
        """
        )

        st.subheader("Test that basic caching works")
        u = my_func(1, 2, dont_care=10)
        # u = my_func(1, 2, dont_care=11)
        v = my_func(1, 2, dont_care=10)
        if u == v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that when you change arguments it's a cache miss")
        v = my_func(10, 2, dont_care=10)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that when you change **kwargs it's a cache miss")
        v = my_func(10, 2, dont_care=100)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that you can turn off caching")
        config.set_option("client.caching", False)
        v = my_func(1, 2, dont_care=10)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that you can turn on caching")
        config.set_option("client.caching", True)


        # Redefine my_func because the st.cache-decorated function "remembers" the
        # config option from when it was declared.
        @st.cache
        def my_func(arg1, arg2=None, *args, **kwargs):
            return random.randint(0, 2 ** 32)


        u = my_func(1, 2, dont_care=10)
        v = my_func(1, 2, dont_care=10)
        if u == v:
            st.success("OK")
        else:
            st.error("Fail")

if demo == "Animation":
    st.empty()
    my_bar = st.progress(0)
    for i in range(100):
        my_bar.progress(i + 1)
        time.sleep(0.1)
    n_elts = int(time.time() * 10) % 5 + 3
    for i in range(n_elts):
        st.text("." * i)
    st.write(n_elts)
    for i in range(n_elts):
        st.text("." * i)
    st.success("done")

if demo == "Code":
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))

if demo == "Plotly example":
    st.title("Plotly examples")

    st.header("Chart with two lines")

    trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
    trace1 = go.Scatter(x=[1, 2, 3, 4], y=[16, 5, 11, 9])
    data = [trace0, trace1]
    st.write(data)

    ###

    st.header("Matplotlib chart in Plotly")

    import matplotlib.pyplot as plt

    f = plt.figure()
    arr = np.random.normal(1, 1, size=100)
    plt.hist(arr, bins=20)

    st.plotly_chart(f)

    ###

    st.header("3D plot")

    x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=12,
            color=z,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )

    data = [trace1]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)

    st.write(fig)

    ###

    st.header("Fancy density plot")

    import plotly.figure_factory as ff

    import numpy as np

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ["Group 1", "Group 2", "Group 3"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    # Plot!
    st.plotly_chart(fig)

if demo == "Reference":
    st.title("Streamlit Quick Reference")

    st.header("The Basics")

    st.write("Import streamlit with `import streamlit as st`.")

    with st.echo():
        st.write(
            """
            The `write` function is Streamlit\'s bread and butter. You can use
            it to write _markdown-formatted_ text in your Streamlit app.
        """
        )

    with st.echo():
        the_meaning_of_life = 40 + 2

        st.write(
            "You can also pass in comma-separated values into `write` just like "
            "with Python's `print`. So you can easily interpolate the values of "
            "variables like this: ",
            the_meaning_of_life,
        )

    st.header("Visualizing data as tables")

    st.write(
        "The `write` function also knows what to do when you pass a NumPy "
        "array or Pandas dataframe."
    )

    with st.echo():
        import numpy as np

        a_random_array = np.random.randn(200, 200)

        st.write("Here's a NumPy example:", a_random_array)

    st.write("And here is a dataframe example:")

    with st.echo():
        import pandas as pd
        from datetime import datetime

        arrays = [
            np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
            np.array(["one", "two", "one", "two", "one", "two", "one", None]),
        ]

        df = pd.DataFrame(
            np.random.randn(8, 4),
            index=arrays,
            columns=[
                datetime(2012, 5, 1),
                datetime(2012, 5, 2),
                datetime(2012, 5, 3),
                datetime(2012, 5, 4),
            ],
        )

        st.write(df, "...and its transpose:", df.T)

    st.header("Visualizing data as charts")

    st.write(
        "Charts are just as simple, but they require us to introduce some "
        "special functions first."
    )

    st.write("So assuming `data_frame` has been defined as...")

    with st.echo():
        chart_data = pd.DataFrame(
            np.random.randn(20, 5), columns=["pv", "uv", "a", "b", "c"]
        )

    st.write("...you can easily draw the charts below:")

    st.subheader("Example of line chart")

    with st.echo():
        st.line_chart(chart_data)

    st.write(
        "As you can see, each column in the dataframe becomes a different "
        "line. Also, values on the _x_ axis are the dataframe's indices. "
        "Which means we can customize them this way:"
    )

    with st.echo():
        chart_data2 = pd.DataFrame(
            np.random.randn(20, 2),
            columns=["stock 1", "stock 2"],
            index=pd.date_range("1/2/2011", periods=20, freq="M"),
        )

        st.line_chart(chart_data2)

    st.subheader("Example of area chart")

    with st.echo():
        st.area_chart(chart_data)

    st.subheader("Example of bar chart")

    with st.echo():
        trimmed_data = chart_data[["pv", "uv"]].iloc[:10]
        st.bar_chart(trimmed_data)

    st.subheader("Matplotlib")

    st.write(
        "You can use Matplotlib in Streamlit. "
        "Just use `st.pyplot()` instead of `plt.show()`."
    )
    try:
        # noqa: F401
        with st.echo():
            from matplotlib import cm, pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # Create some data
            X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
            Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

            # Plot the surface.
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

            st.pyplot()
    except Exception as e:
        err_str = str(e)
        if err_str.startswith("Python is not installed as a framework."):
            err_str = (
                "Matplotlib backend is not compatible with your Python "
                'installation. Please consider adding "backend: TkAgg" to your '
                " ~/.matplitlib/matplotlibrc. For more information, please see "
                '"Working with Matplotlib on OSX" in the Matplotlib FAQ.'
            )
        st.warning("Error running matplotlib: " + err_str)

    st.subheader("Vega-Lite")

    st.write(
        "For complex interactive charts, you can use "
        "[Vega-Lite](https://vega.github.io/vega-lite/):"
    )

    with st.echo():
        df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])

        st.vega_lite_chart(
            df,
            {
                "mark": "circle",
                "encoding": {
                    "x": {"field": "a", "type": "quantitative"},
                    "y": {"field": "b", "type": "quantitative"},
                    "size": {"field": "c", "type": "quantitative"},
                    "color": {"field": "c", "type": "quantitative"},
                },
                # Add zooming/panning:
                "selection": {"grid": {"type": "interval", "bind": "scales"}},
            },
        )

    st.header("Visualizing data as images via Pillow.")


    @st.cache(persist=True)
    def read_file_from_url(url):
        try:
            return requests.get(url).content
        except requests.exceptions.RequestException:
            st.error("Unable to load file from %s. " "Is the internet connected?" % url)
        except Exception as e:
            st.exception(e)
        return None


    image_url = (
        "https://images.fineartamerica.com/images/artworkimages/"
        "mediumlarge/1/serene-sunset-robert-bynum.jpg"
    )
    image_bytes = read_file_from_url(image_url)

    if image_bytes is not None:
        with st.echo():
            # We can pass URLs to st.image:
            st.image(image_url, caption="Sunset", use_column_width=True)

            # For some reason, `PIL` requires you to import `Image` this way.
            from PIL import Image

            image = Image.open(BytesIO(image_bytes))

            array = np.array(image).transpose((2, 0, 1))
            channels = array.reshape(array.shape + (1,))

            # st.image also accepts byte arrays:
            st.image(channels, caption=["Red", "Green", "Blue"], width=200)

    st.header("Visualizing data as images via OpenCV")

    st.write("Streamlit also supports OpenCV!")
    try:
        import cv2

        if image_bytes is not None:
            with st.echo():
                image = cv2.cvtColor(
                    cv2.imdecode(np.fromstring(image_bytes, dtype="uint8"), 1),
                    cv2.COLOR_BGR2RGB,
                )

                st.image(image, caption="Sunset", use_column_width=True)
                st.image(cv2.split(image), caption=["Red", "Green", "Blue"], width=200)
    except ImportError as e:
        st.write(
            "If you install opencv with the command `pip install opencv-python-headless` "
            "this section will tell you how to use it."
        )

        st.warning("Error running opencv: " + str(e))

    st.header("Inserting headers")

    st.write(
        "To insert titles and headers like the ones on this page, use the `title`, "
        "`header`, and `subheader` functions."
    )

    st.header("Preformatted text")

    with st.echo():
        st.text(
            "Here's preformatted text instead of _Markdown_!\n"
            "       ^^^^^^^^^^^^\n"
            "Rock on! \m/(^_^)\m/ "
        )

    st.header("JSON")

    with st.echo():
        st.json({"hello": "world"})

    with st.echo():
        st.json('{"object":{"array":[1,true,"3"]}}')

    st.header("Inline Code Blocks")

    with st.echo():
        with st.echo():
            st.write("Use `st.echo()` to display inline code blocks.")

    st.header("Alert boxes")

    with st.echo():
        st.error("This is an error message")
        st.warning("This is a warning message")
        st.info("This is an info message")
        st.success("This is a success message")

    st.header("Progress Bars")

    with st.echo():
        for percent in [0, 25, 50, 75, 100]:
            st.write("%s%% progress:" % percent)
            st.progress(percent)

    st.header("Help")

    with st.echo():
        st.help(dir)

    st.header("Out-of-Order Writing")

    st.write("Placeholders allow you to draw items out-of-order. For example:")

    with st.echo():
        st.text("A")
        placeholder = st.empty()
        st.text("C")
        placeholder.text("B")

    st.header("Exceptions")
    st.write("You can print out exceptions using `st.exception()`:")

    with st.echo():
        try:
            raise RuntimeError("An exception")
        except Exception as e:
            st.exception(e)

    st.header("Playing audio")

    audio_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c4/"
        "Muriel-Nguyen-Xuan-Chopin-valse-opus64-1.ogg"
    )
    audio_bytes = read_file_from_url(audio_url)

    st.write(
        """
        Streamlit can play audio in all formats supported by modern
        browsers. Below is an example of an _ogg_-formatted file:
        """
    )

    if audio_bytes is not None:
        with st.echo():
            st.audio(audio_bytes, format="audio/ogg")

    st.header("Playing video")

    st.write(
        """
        Streamlit can play video in all formats supported by modern
        browsers. Below is an example of an _mp4_-formatted file:
        """
    )

    video_url = "https://archive.org/download/WildlifeSampleVideo/" "Wildlife.mp4"
    video_bytes = read_file_from_url(video_url)

    if video_bytes is not None:
        with st.echo():
            st.video(video_bytes, format="video/mp4")

    st.header("Lengthy Computations")
    st.write(
        """
        If you're repeatedly running length computations, try caching the
        solution.
        ```python
        @streamlit.cache
        def lengthy_computation(...):
            ...
        # This runs quickly.
        answer = lengthy_computation(...)
        ```
        **Note**: `@streamlit.cache` requires that the function output
        depends *only* on its input arguments. For example, you can cache
        calls to API endpoints, but only do so if the data you get won't change.
    """
    )
    st.subheader("Spinners")
    st.write("A visual way of showing long computation is with a spinner:")


    def lengthy_computation():
        pass  # noop for demsontration purposes.


    with st.echo():
        with st.spinner("Computing something time consuming..."):
            lengthy_computation()

    st.header("Animation")
    st.write(
        """
        Every Streamlit method (except `st.write`) returns a handle
        which can be used for animation. Just call your favorite
        Streamlit function (e.g. `st.xyz()`) on the handle (e.g. `handle.xyz()`)
        and it will update that point in the app.
        Additionally, you can use `add_rows()` to append numpy arrays or
        DataFrames to existing elements.
    """
    )

    with st.echo():
        import numpy as np
        import time

        bar = st.progress(0)
        complete = st.text("0% complete")
        graph = st.line_chart()

        for i in range(100):
            bar.progress(i + 1)
            complete.text("%i%% complete" % (i + 1))
            graph.add_rows(np.random.randn(1, 2))

            time.sleep(0.1)

if demo == "Run on save":
    st.title("Test of run-on-save")
    secs_to_wait = 5

    """
    How to test this:
    """

    st.info(
        """
        **First of all, make sure you're running the dev version of Streamlit** or
        that this file lives outside the Streamlit distribution. Otherwise, changes
        to this file may be ignored!
    """
    )

    """
    1. If run-on-save is on, make sure the page changes every few seconds. Then
       turn run-on-save off in the settigns menu and check (2).
    2. If run-on-save is off, make sure "Rerun"/"Always rerun" buttons appear in
       the status area. Click "Always rerun" and check (1).
    """

    st.write("This should change every ", secs_to_wait, " seconds: ", random())

    # Sleep for 5s (rather than, say, 1s) because on the first run we need to make
    # sure Streamlit is fully initialized before the timer below expires. This can
    # take several seconds.
    status = st.empty()
    for i in range(secs_to_wait, 0, -1):
        time.sleep(1)
        status.text("Sleeping %ss..." % i)

    status.text("Touching %s" % __file__)

    platform_system = platform.system()

    if platform_system == "Linux":
        cmd = (
                "sed -i "
                "'s/^# MODIFIED AT:.*/# MODIFIED AT: %(time)s/' %(file)s"
                " && touch %(file)s"
                % {  # sed on Linux modifies a different file.
                    "time": time.time(),
                    "file": __file__,
                }
        )

    elif platform_system == "Darwin":
        cmd = "sed -i bak " "'s/^# MODIFIED AT:.*/# MODIFIED AT: %s/' %s" % (
            time.time(),
            __file__,
        )

    # elif platform_system == "Windows":
    #     raise NotImplementedError("Windows not supported")
    #
    # else:
    #     raise Exception("Unknown platform")
    #
    # os.system(cmd)
    #
    # status.text("Touched %s" % __file__)

    # MODIFIED AT: 1580332945.720056

if demo == "Syntax error":
    st.title("Syntax error test")

    st.info("Uncomment the comment blocks in the source code one at a time.")

    st.write(
        """
        Here's the source file for you to edit:
        ```
        examples/syntax_error.py
        ```
        """
    )

    st.write("(Some top text)")

    # # Uncomment this as a block.
    # a = not_a_real_variable  # EXPECTED: inline exception.

    # # Uncomment this as a block.
    # if True  # EXPECTED: modal dialog

    # # Uncomment this as a block.
    # sys.stderr.write('Hello!\n')  # You should not see this.
    # # The line below is a compile-time error. Bad indentation.
    #        this_indentation_is_wrong = True  # EXPECTED: modal dialog

    # # Uncomment this as a block.
    # # This tests that errors after the first st call get caught.
    # a = not_a_real_variable  # EXPECTED: inline exception.

    st.write("(Some bottom text)")

if demo == "Syntax hilite":
    Language = namedtuple("Language", ["name", "example"])

    languages = [
        Language(
            name="Python",
            example="""
    # Python
    def say_hello():
        name = 'Streamlit'
        print('Hello, %s!' % name)""",
        ),
        Language(
            name="C",
            example="""
    /* C */
    int main(void) {
        const char *name = "Streamlit";
        printf(\"Hello, %s!\", name);
        return 0;
    }""",
        ),
        Language(
            name="JavaScript",
            example="""
    /* JavaScript */
    function sayHello() {
        const name = 'Streamlit';
        console.log(`Hello, ${name}!`);
    }""",
        ),
        Language(
            name="Shell",
            example="""
    # Bash/Shell
    NAME="Streamlit"
    echo "Hello, ${NAME}!"
    """,
        ),
        Language(
            name="SQL",
            example="""
    /* SQL */
    SELECT * FROM software WHERE name = 'Streamlit';
    """,
        ),
        Language(
            name="JSON",
            example="""
    {
        "_comment": "This is a JSON file!",
        name: "Streamlit",
        version: 0.27
    }""",
        ),
        Language(
            name="YAML",
            example="""
    # YAML
    software:
        name: Streamlit
        version: 0.27
    """,
        ),
        Language(
            name="HTML",
            example="""
    <!-- HTML -->
    <head>
      <title>Hello, Streamlit!</title>
    </head>
    """,
        ),
        Language(
            name="CSS",
            example="""
    /* CSS */
    .style .token.string {
        color: #9a6e3a;
        background: hsla(0, 0%, 100%, .5);
    }
    """,
        ),
        Language(
            name="JavaScript",
            example="""
    console.log('This is an extremely looooooooooooooooooooooooooooooooooooooooooooooooooooong string.')
        """,
        ),
    ]

    st.header("Syntax hiliting")

    st.subheader("Languages")
    for lang in languages:
        st.code(lang.example, lang.name)

    st.subheader("Other stuff")
    with st.echo():
        print("I'm inside an st.echo() block!")

    st.markdown(
        """
    This is a _markdown_ block...
    ```python
    print('...and syntax hiliting works here, too')
    ```
    """
    )

if demo == "Video":
    VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]

    # For sample video files, try the Internet Archive, or download a few samples here:
    # http://techslides.com/sample-webm-ogg-and-mp4-video-files-for-html5

    st.title("Video Widget Examples")

    st.header("Local video files")
    st.write(
        "You can use st.video to play a locally-stored video by supplying it with a valid filesystem path."
    )


    def get_video_files_in_dir(directory):
        out = []
        for item in os.listdir(directory):
            try:
                name, ext = item.split(".")
            except:
                continue
            if name and ext:
                if ext in VIDEO_EXTENSIONS:
                    out.append(item)
        return out


    avdir = os.path.expanduser("~")
    files = get_video_files_in_dir(avdir)

    if len(files) == 0:
        st.write(
            "Put some video files in your home directory (%s) to activate this player."
            % avdir
        )

    else:
        filename = st.selectbox(
            "Select a video file from your home directory (%s) to play" % avdir, files, 0,
        )

        st.video(os.path.join(avdir, filename))
    st.header("Remote video playback")
    st.write("st.video allows a variety of HTML5 supported video links, including YouTube.")


    def shorten_vid_option(opt):
        return opt.split("/")[-1]


    # A random sampling of videos found around the web.  We should replace
    # these with those sourced from the streamlit community if possible!
    vidurl = st.selectbox(
        "Pick a video to play",
        (
            "https://youtu.be/_T8LGqJtuGc",
            "https://www.youtube.com/watch?v=kmfC-i9WgH0",
            "https://www.youtube.com/embed/sSn4e1lLVpA",
            "http://www.rochikahn.com/video/videos/zapatillas.mp4",
            "http://www.marmosetcare.com/video/in-the-wild/intro.webm",
            "https://www.orthopedicone.com/u/home-vid-4.mp4",
        ),
        0,
        shorten_vid_option,
    )

    st.video(vidurl)
# if st.checkbox('Show Raw Data'):
#     '## Raw Data at %sh' % hour,data
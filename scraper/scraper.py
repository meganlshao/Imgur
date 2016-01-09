# scraper.py, created by Megan Shao (mshao@hmc.edu) and Vincent Fiorentini
# (vfiorentini@hmc.edu).
# Collects and saves comment information from Imgur.
# To use, replace the client_id and client_secret in getClient() and run.
# When run, this file will collect all the comments from the most recent 
# numGalleryPages days of most popular posts from Imgur. The results will be
# saved in a file with a name of the form "commentFeaturesList_timestamp.csv".
# This file requires you to have ImgurPython installed and an Imgur client.

from imgurpython import ImgurClient

import matplotlib.pyplot as plt
import numpy as np
import csv
import time

# For reference, the Imgur API can be found here: https://api.imgur.com/
# For reference, the Imgur Python client can be found here: https://github.com/Imgur/imgurpython
def getClient():
    """Returns an ImgurClient for data collection."""
    client_id = 'your_client_id'
    client_secret = 'your_client_secret'
    client = ImgurClient(client_id, client_secret)
    return client

def numCommentsInComment(comment):
    """Given a comment, returns the number of comments descended from this comment.
       Includes this comment in the count."""
    return 1 + sum(map(numCommentsInComment, comment.children))

def flattenCommentChildren(comment):
    """Given a comment, returns a list of all the comments descended from this comment.
       Includes this comment in the list."""
    comments = [comment]
    for child in comment.children:
        comments.extend(flattenCommentChildren(child))
    return comments

def flattenCommentList(comments):
    """Given a list of comments, returns a list of all the comments descended from these comments.
       Includes these comments in the list."""
    allComments = []
    for topLevelComment in comments:
        allComments.extend(flattenCommentChildren(topLevelComment))
    return allComments

def getUpvoteDifference(comment):
    """Given a comment, returns the difference between its upvotes and downvotes."""
    return comment.ups - comment.downs

def visualizeUpvoteData(comments, metric, metricname, bins):
    """Given a list of comments and a function to compute the metric of interest,
       plots a histogram of each comment's ratio of upvotes to all votes."""
    ratios = map(metric, comments)
    plt.hist(ratios, bins=bins)
    plt.ylabel('number of comments')
    plt.xlabel(metricname)
    plt.show()

def convertToString(text):
    """Converts the given text to utf8."""
    return text.encode('utf8', 'ignore')

class CommentFeature:
    """Represents a comment as its important pieces of data.
       This object is not itself a real-valued feature vector."""

    def __init__(self, comment, galleryItem):
        """Get and store all pieces of information that might be processed
           into the final feature vector."""
        # The following are the relevant pieces of information:
        # the comment itself, time difference between original post and comment,
        # whether the comment is a reply, upvotes, downvotes, net upvotes,
        # original post title, original post time posted, original post 
        # description, original post category, original post net upvotes

        # the comment itself (string)
        # text can have weird characters
        self.text = convertToString(comment.comment)

        # is comment reply (bool)
        if comment.parent_id:
            self.isReply = True
        else:
            self.isReply = False

        # whether the comment author is the Original Poster of the image (bool)
        self.isAuthorOP = comment.author_id == galleryItem.account_id

        # time difference between original post and comment - in seconds (int)
        # (how long after the gallery item was posted was this comment  posted)
        self.datetimeDelta = comment.datetime - galleryItem.datetime

        # upvotes (int)
        self.upvotes = comment.ups
        # downvotes (int)
        self.downvotes = comment.downs
        # net upvotes (int)
        self.netUpvotes = comment.ups - comment.downs
        
        # [original post] title (string)
        self.postTitle = convertToString(galleryItem.title)
        # [original post] time posted - in seconds, epoch time (int)
        self.postTime = galleryItem.datetime
        # [original post] description (string)
        postDescription = galleryItem.description
        if postDescription:
            self.postDescription = convertToString(galleryItem.description)
        else:
            self.postDescription = ""
        # [original post] category (string)
        self.postCategory = convertToString(galleryItem.section)
        # [original post] net upvotes
        self.postNetUpvotes = galleryItem.ups - galleryItem.downs
        

    def asDataList(self):
        """Returns a list representing the properties of this comment.
           This method is useful, for example, in storing this object in a CSV.
        """
        # NOTE: Make sure to keep this method in-sync with asDataListLabels().
        return [self.text, self.isReply, 
                self.isAuthorOP,
                self.datetimeDelta, self.upvotes, self.downvotes, self.netUpvotes,
                self.postTitle, self.postTime, self.postDescription, self.postCategory,
                self.postNetUpvotes]
    def asDataListLabels(self):
        """Returns a list indicating the fields of asDataList()."""
        # NOTE: Make sure to keep this method in-sync with asDataList().
        return ["text", "isReply",
                "isAuthorOP",
                "datetimeDelta", "upvotes", "downvotes", "netUpvotes",
                "postTitle", "postTime", "postDescription", "postCategory",
                "postNetUpvotes"]

    def __repr__(self):
        return "{" + ",".join(map(str, self.asDataList())) + "}"


def commentListToFeatureList(commentList, galleryItem):
    """Given a list of comments, returns a list of corresponding features."""
    featureList = map(lambda comment: CommentFeature(comment, galleryItem), commentList)
    return featureList

def commentListToCommentFeatures(commentList, galleryItem):
    """Given a list of comments, returns a list of corresponding CommentFeatures.
       Given a list of comments, a gallery item, and an API client."""
    commentFeatures = []
    for comment in commentList:
        commentFeature = CommentFeature(comment, galleryItem)
        commentFeatures.append(commentFeature)
    return commentFeatures

def main():
    numGalleryPages = 1 # number of pages (i.e., days) of galleries to get

    limitRate = True
    itemSleepTime = 5 # seconds to sleep after each one item
    itemFailSleepTime = 75 # seconds to sleep after failing an item
    maxRetriesPerItem = 10 # maximum number of times to try each call that might fail
    maxFails = 1000 # global maximum number of failures to allow (i.e., to stop trying after hitting rate limit)

    visualizeData = False # plot a histogram of the first result instead of collecting all the data

    client = getClient()

    # we will construct a list of data lists (which also contain the labels)
    commentFeaturesList = []

    print "=================================================="
    print "Will look through %d pages of images" % numGalleryPages

    numFails = 0
    for pageNumber in range(0, numGalleryPages):

        galleryItems = None
        for tryNum in range(maxRetriesPerItem):
            try:
                # get the top images for today
                print "--------------------------------------------------"
                print "Getting page number %d of gallery items for hot, viral daily images." % pageNumber
                galleryItems = client.gallery(section='hot', sort='viral', page=pageNumber, window='day',
                                              show_viral=True)
                print "Got %d gallery items to look through." % len(galleryItems)
                break
            except:
                numFails += 1
                if numFails > maxFails:
                    raise Exception("Reached %d fails, giving up." % maxFails)
                print("Gallery error - maybe internet disconnected? Retrying (number %d)" % tryNum)
                print("(But first, sleeping for %d seconds)" % itemFailSleepTime)
                time.sleep(itemFailSleepTime)

        if visualizeData:
            galleryItems = galleryItems[:1]
        
        try:
            for item in galleryItems:
                for tryNum in range(maxRetriesPerItem):
                    try:
                        # collect comments
                        topLevelComments = client.gallery_item_comments(item.id, sort='best')
                        allComments = flattenCommentList(topLevelComments)
                        # print statistics
                        print ("The gallery at %s has %d top-level comments and %d total comments." 
                            % (item.link, len(topLevelComments), len(allComments)))

                        # display comment popularity distribution
                        if visualizeData:
                            plt.title('all comments'); visualizeUpvoteData(allComments, getUpvoteDifference, 
                                                                           "net upvotes",
                                                                           np.arange(-5.5, 10.5, 1))

                        # get data list for all the commentList
                        commentFeatures = commentListToCommentFeatures(allComments, item) 
                        commentFeaturesList.extend(commentFeatures)

                        if limitRate:
                            print("Sleeping for %d seconds" % itemSleepTime)
                            time.sleep(itemSleepTime)
                        break
                    except:
                        numFails += 1
                        if numFails > maxFails:
                            raise Exception("Reached %d fails, giving up." % maxFails)
                        print("Error - maybe internet disconnected? Retrying (number %d)" % tryNum)
                        print("(But first, sleeping for %d seconds)" % itemFailSleepTime)
                        time.sleep(itemFailSleepTime)
        except:
            print "Error - probably hit rate limit. Writing *partial results* to file anyway."
            break

    print ("Finished extracting data. There are %d example comments from %d gallery items." 
        % (len(commentFeaturesList), len(galleryItems)))

    if not visualizeData:
        # save our data list to a CSV file
        outputFilename = "commentFeaturesList_" + str(int(time.time())) + ".csv"
        with open(outputFilename, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            # write the data list labels as the first row
            writer.writerow(commentFeaturesList[0].asDataListLabels())
            for commentFeature in commentFeaturesList:
                writer.writerow(commentFeature.asDataList())
        print "Saved data list to %s" % outputFilename

if __name__ == "__main__":
    main()
